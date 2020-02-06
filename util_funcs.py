from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os, json
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from data_utils import InputExample, InputFeatures
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

logger = logging.getLogger('root')

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def train(args, train_dataset, model, params_to_update = "ALL", evaluate_during_training = True,patience = 1e5, eval_dataset = None,\
     pred_genre = False, pred_both = False, multiple_eval_datasets = None, dev_dataset = None, multi_task = False):
    """ Train the model """
    """ 
    Specify which parameters to update. Three strategies, "ALL" means fine-tune the whole model, 
    "Classifier" means keeping all the representation fixed, only train the parameters of the classifier.
    "CLS" means updating the pooling layer and the classifier? 
    """
    logger = logging.getLogger("root")
    final_results = {}
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    es = EarlyStopping(patience=patience)
    es_flag = False
    if args.max_steps > 0:
        t_total = min(args.max_steps,  args.num_train_epochs *(len(train_dataloader)* args.gradient_accumulation_steps))
        logging_steps = min(args.logging_steps, t_total//2)
        #args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if params_to_update == "ALL":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        logger.info("Updated params are: %s", "All" )
    elif params_to_update == "Classifier":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)\
                and ( "classifier" in n or "pooler" in n ) ], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)\
                and ( "classifier" in n or "pooler" in n )], 'weight_decay': 0.0}
            ]
        logger.info("Updated params are: %s", [n for n, p in model.named_parameters() if  "classifier" in n or "pooler" in n] )
        

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate_bert, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if not multi_task and not pred_both:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]
            elif pred_both:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3] + batch[4] * 2}
                outputs = model(**inputs)
                loss = outputs[0]
            else:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels_1':batch[3],  'labels_2':batch[4],}
                outputs = model(**inputs)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % 20 == 0:
                logger.info("the %d step :, %s", global_step, loss)
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and evaluate_during_training and (global_step % logging_steps == 0):
                    # Log metrics
                    # if multiple_eval_datasets is not None:
                    #     final_results[global_step] = []
                    #     for eval_dataset in multiple_eval_datasets:
                    #         results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
                    #         final_results[global_step].append(results['acc'])
                    #         logger.info("the %d step :, %s", global_step, results['acc'])

                    # else:
                    #     results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
                    #     logger.info("the %d step :, %s, %s, eval loss : %s, training loss : %s", global_step, results['acc'], results['macro_f1'], ev_loss , tr_loss/logging_steps)
                    #     final_results[global_step] = (results['acc'])
                    
                    dev_acc, ev_loss = evaluate(args = args, eval_dataset = dev_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
                    if multi_task:
                        #only look at the sentiment accuracy
                        dev_acc = dev_acc[0]
                    logger.warning("the %d step :, dev acc: %s, eval loss : %s, training loss : %s", global_step, dev_acc, ev_loss , tr_loss/logging_steps)
                    es_flag = es.step(dev_acc)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if multi_task:
                        model_to_save = model.module.bert if hasattr(model, 'module') else model.bert 
                    else:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if (args.max_steps > 0 and global_step > args.max_steps) or es_flag:
                epoch_iterator.close()
                break
        if (args.max_steps > 0 and global_step > args.max_steps) or es_flag:
            train_iterator.close()
            break
    results, ev_loss = evaluate(args = args, eval_dataset = eval_dataset,  model = model, pred_both = pred_both, multi_task=multi_task)
    final_results[global_step].append(results['acc'])
    logger.info("the %d step :, %s", global_step, results['acc'])
    logger.warning('########final_results############')
    logger.warning(final_results)
    with open(os.path.join(args.output_dir, "finetune_final_results.json"),"w") as f:
        f.write(json.dumps(final_results))
    final_results = json.load(open(os.path.join(args.output_dir, 'finetune_final_results.json')))
    # plot_results_Bert(final_results, args.output_dir, show = False, pred_both=pred_both)
    return global_step, tr_loss / global_step

class MTBert(nn.Module):
    def __init__(self, bert_model):
        super(MTBert, self).__init__()
        self.bert = bert_model
        config = self.bert.config
        self.classifier_1 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels_1=None, labels_2 = None, head_mask=None):
        outputs = self.bert.bert(input_ids, token_type_ids, attention_mask)

        _, pooled_output = outputs
        pooled_output = self.bert.dropout(pooled_output)
        logits_1 = self.classifier_1(pooled_output)
        logits_2 = self.classifier_2(pooled_output)

        if labels_1 is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits_1.view(-1, self.bert.config.num_labels), labels_1.view(-1)) + loss_fct(logits_2.view(-1, self.bert.config.num_labels), labels_2.view(-1),)
            return loss,logits_1,logits_2
        return logits_1,logits_2


def evaluate(args, eval_dataset, model, prefix="", pred_genre = False, pred_both = False, multi_task = False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    
    eval_output_dir = args.output_dir
    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_1 = None
    preds_2 = None
    out_label_ids_1 = None
    out_label_ids_2 = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if pred_both:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3] + 2 * batch[4]}
            elif multi_task:
                inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels_1': batch[3],
                            'labels_2': batch[4],}
            else:
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
            outputs = model(**inputs)
            if multi_task:
                tmp_eval_loss, logits_1, logits_2 = outputs[:3]
            else:
                tmp_eval_loss, logits_1 = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds_1 is None:
            preds_1 = logits_1.detach().cpu().numpy()
            if multi_task:
                out_label_ids_1 = inputs['labels_1'].detach().cpu().numpy()
            else:
                out_label_ids_1 = inputs['labels'].detach().cpu().numpy()

        else:
            preds_1 = np.append(preds_1, logits_1.detach().cpu().numpy(), axis=0)
            if multi_task:
                out_label_ids_1 = np.append(out_label_ids_1, inputs['labels_1'].detach().cpu().numpy(), axis=0)
            else:
                out_label_ids_1 = np.append(out_label_ids_1, inputs['labels'].detach().cpu().numpy(), axis=0)

        if multi_task:
            if preds_2 is None:
                preds_2 = logits_2.detach().cpu().numpy()
                out_label_ids_2 = inputs['labels_2'].detach().cpu().numpy()
            else:
                preds_2 = np.append(preds_2, logits_2.detach().cpu().numpy(), axis=0)
                out_label_ids_2 = np.append(out_label_ids_2, inputs['labels_2'].detach().cpu().numpy(), axis=0)


    eval_loss = eval_loss / nb_eval_steps
    preds_1 = np.argmax(preds_1, axis=1)
    if multi_task:
        preds_2 = np.argmax(preds_2, axis=1)

    if pred_both:
        acc_s = (preds_1 % 2 == out_label_ids_1 % 2).mean()
        acc_g = ((preds_1 > 1) == (out_label_ids_1 > 1)).mean()
        acc = (preds_1 == out_label_ids_1).mean()
        accuracy = (acc_s, acc_g, acc)
    elif multi_task:
        acc_1 = (preds_1 == out_label_ids_1).mean()
        acc_2 = (preds_2 == out_label_ids_2).mean()
        accuracy = (acc_1, acc_2)
    else:
        accuracy = (preds_1 == out_label_ids_1).mean()


    macro_f1 = f1_score(y_true=out_label_ids_1, y_pred=preds_1, average="macro")

    # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results {} *****".format(prefix))
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
    #wandb.log({'ev/acc': results['acc'],'ev/macro_f1': results['macro_f1']})
    return accuracy,eval_loss



def calc_emb(dataset, model,model_type, device):
    all_embs = [None for _ in range(3)] if model_type != 'Bert' else None
    train_dataloader = DataLoader(dataset, sampler=None, batch_size=8)
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            #logits_1 = self.classifier(cls_embs)
            if model_type == 'AdvBert':
                batch = tuple(t.to(device) for t in batch)
                common_embs =  model.common_encoder(batch[0])
                sent_embs = model.encoder_sent(common_embs)
                other_embs = model.encoder_other(common_embs)
            elif model_type == 'TripletBert':
                batch = [[t.to(device) for t in trip] for trip in batch]
                sent_embs = model.encoder_sent.mlp(batch[0][0])
                other_embs = model.encoder_other.mlp(batch[0][0])
            elif model_type == 'TripletBertEDA':
                batch = tuple(t.to(device) for t in batch)
                sent_embs = model.encoder_sent.mlp(batch[0])
                other_embs = model.encoder_other.mlp(batch[0])
            elif model_type == 'Bert':
                batch = tuple(t.to(device) for t in batch)
                if  hasattr(model, 'module'):
                    model = model.module
                sent_embs = model.bert(input_ids = batch[0],
                                attention_mask=batch[1],
                                token_type_ids=batch[2])[1]
            if model_type != 'Bert':
                combined_embs = torch.cat((sent_embs,other_embs), dim=1)
                for i,embs in enumerate([sent_embs, other_embs, combined_embs]):
                    if all_embs[i] is None:
                        all_embs[i] = embs.detach().cpu().numpy()
                    else:
                        all_embs[i] = np.concatenate((all_embs[i], embs.detach().cpu().numpy()), axis=0)
            else:
                if all_embs is None:
                    all_embs = sent_embs.detach().cpu().numpy()
                else:
                    all_embs = np.concatenate((all_embs, sent_embs.detach().cpu().numpy()), axis=0)

    return all_embs

def lr_eval(train_embs, eval_embs,train_labels,eval_labels):

    normalizer = Normalizer()
    train_embs = normalizer.fit_transform(train_embs) 
    eval_embs = normalizer.transform(eval_embs)
    lr_model = LogisticRegression(random_state=0, penalty='l2', solver = 'liblinear')


    ##drop all negative labels
    non_neg = [i for i in range(len(train_labels)) if train_labels[i] >= 0  ]
    if len(non_neg) == 0:
        return 0,0
    else:
        train_embs = [train_embs[i] for i in non_neg]
        train_labels = [train_labels[i] for i in non_neg]
    num_classes = len(list(set(train_labels)))
    if num_classes == 1:
        return 0,0
    elif num_classes > 2:
        logger.warning('3 classes, something is wrong')
    lr_model.fit(X = train_embs, y = train_labels)
    y_pred = lr_model.predict(eval_embs)
    acc = sum(y_pred == eval_labels)/len(y_pred)
    weights = lr_model.coef_[0]
    dim = int(len(weights) / 2)
    weght_ratio = np.linalg.norm(weights[:dim])/ np.linalg.norm(weights[dim:])

    return acc, weght_ratio

def lr_eval_Bert(Bert_model, train_dataset, eval_datasets, device):
    all_results = []
    all_train_embs = calc_emb(dataset = train_dataset, model = Bert_model,model_type = 'Bert', device = device)
    train_sent_labels = [d[-2].item() for d in train_dataset]
    train_genre_labels = [d[-1].item() for d in train_dataset]
    for eval_dataset in eval_datasets:
        eval_sent_labels = [d[-2].item() for d in eval_dataset]
        eval_genre_labels = [d[-1].item() for d in eval_dataset]

        all_eval_embs = calc_emb(dataset = eval_dataset, model = Bert_model,model_type = 'Bert', device = device)
        i = 0
        acc_sent, weght_ratio_sent = lr_eval(all_train_embs[i], all_eval_embs[i],train_labels = train_sent_labels,eval_labels= eval_sent_labels)
        acc_genre, weght_ratio_genre = lr_eval(all_train_embs[i], all_eval_embs[i],train_labels = train_genre_labels,eval_labels= eval_genre_labels)
        print ('acc_sent: ', acc_sent)
        print ('acc_genre: ', acc_genre)
        all_results.append([acc_sent, acc_genre])
    return all_results
    
def lr_eval_DEBert(DEBert_model,model_type,  train_emb_dataset, all_eval_emb_datasets, device):
    all_results = {}
    all_train_embs = calc_emb(dataset = train_emb_dataset, model = DEBert_model,model_type = model_type, device = device)
    if model_type == 'TripletBert':
        train_sent_labels = [d[0][-2].item() for d in train_emb_dataset]
        train_genre_labels = [d[0][-1].item() for d in train_emb_dataset]
    else:
        train_sent_labels = [d[-2].item() for d in train_emb_dataset]
        train_genre_labels = [d[-1].item() for d in train_emb_dataset]
    for ratio_id,eval_dataset in enumerate(all_eval_emb_datasets):
        all_results[ratio_id] = {}
        if model_type == 'TripletBert':
            eval_sent_labels = [d[0][-2].item() for d in eval_dataset]
            eval_genre_labels = [d[0][-1].item() for d in eval_dataset]
        else:
            eval_sent_labels = [d[-2].item() for d in eval_dataset]
            eval_genre_labels = [d[-1].item() for d in eval_dataset]
        all_eval_embs = calc_emb(dataset = eval_dataset, model = DEBert_model,model_type = model_type, device = device)
        for i,emb_type in enumerate(['sent_enc', 'other_enc', 'combined']):
            acc_sent, weght_ratio_sent = lr_eval(all_train_embs[i], all_eval_embs[i],train_labels = train_sent_labels,eval_labels= eval_sent_labels)
            acc_genre, weght_ratio_genre = lr_eval(all_train_embs[i], all_eval_embs[i],train_labels = train_genre_labels,eval_labels= eval_genre_labels)
#             print ('acc_sent: ', acc_sent)
#             print ('acc_genre: ', acc_genre)
            all_results[ratio_id][emb_type] = ((acc_sent, acc_genre), (weght_ratio_sent, weght_ratio_genre))
            
    return all_results

