
import os
import logging
import argparse
import datetime
import json
import random
from tqdm import tqdm, trange

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset,Dataset)
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from data_utils import InputExample, InputFeatures, load_dataset
from util_funcs import train, evaluate, calc_emb, lr_eval_Bert, lr_eval_DEBert,EarlyStopping


"""
Disentangle a pre-trained model using either triplet loss or some VAE or a combination of the two.
Takes the original model as input, return the disentangled new model.
"""


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

class TripletBert(nn.Module):
    def __init__(self, margin_sent = 0.0, margin_other = 0.0, input_size = 768,\
        hidden_size = 768, output_size = 768, beta_recon = 0, beta_other = 1):
        super(TripletBert, self).__init__()
        self.output_size = output_size
        self.margin_sent = margin_sent
        self.margin_other = margin_other
        self.encoder_sent = Triplet_Encoder(input_size= input_size,\
             hidden_size = hidden_size, output_size = output_size, margin = margin_sent)
        self.encoder_other = Triplet_Encoder(input_size= input_size, \
            hidden_size = hidden_size, output_size = output_size, margin = margin_other)

        self.decoder = nn.Sequential(nn.Linear(2 * output_size, 4 * output_size),\
             nn.Tanh(), nn.Linear(4 * output_size, input_size))
        self.beta_recon = beta_recon
        self.beta_other = beta_other


    def forward(self, input_embs):
        loss_sent = self.encoder_sent(input_embs[0], input_embs[1], input_embs[2])
        loss_other = self.encoder_other(input_embs[0], input_embs[2], input_embs[1])
        loss_recon = 0
        for emb in input_embs:
            sent_output = self.encoder_sent.mlp(emb)
            other_output = self.encoder_other.mlp(emb)
            recon = self.decoder(torch.cat((sent_output, other_output), 1))
            loss_recon += torch.norm(recon - emb, 2)
        
        #reconstruction loss is the average of the three embeddings
        loss_recon = loss_recon / 3
        loss = loss_sent + self.beta_other * loss_other + self.beta_recon * loss_recon

        return loss,loss_sent, loss_other,loss_recon



class Triplet_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, margin):
        super(Triplet_Encoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size),\
             nn.Tanh(),nn.Linear(hidden_size, hidden_size),nn.Tanh(), nn.Linear(hidden_size, output_size))
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    def forward(self, anchor, positive, negative):
        anchor = self.mlp(anchor)
        positive = self.mlp(positive)
        negative = self.mlp(negative)

        loss = self.loss_fn(anchor, positive, negative)
        return loss


class TripletDataset(Dataset):

    def __init__(self, emb_dataset, random_neg = False, strategy = 'opposite', raw_sents = None):
        #For now assume we have two labels and two tasks
        super(TripletDataset, self).__init__()
        self.emb_dataset = emb_dataset
        self.num_data = len(emb_dataset)
        self.labels = np.asanyarray([d[-2].item() for d in emb_dataset])
        self.genres = np.asanyarray([d[-1].item() for d in emb_dataset])
        self.random_neg = random_neg
        self.strategy = strategy

        #if using other embedding to generate the triplets, need to load the embeddings when initializing
        if self.strategy in ['bert_distance','sts_distance','nli_distance']:
            #load raw sents
            pass


        # load the bert distance matrix      
    def __getitem__(self, index):

        ###TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        anchor_data = self.emb_dataset[index]
        label = self.emb_dataset[index][-2].item()
        genre = self.emb_dataset[index][-1].item()

        if self.strategy == 'opposite':
            same_l_candidates_inds =[i for i in range(self.num_data) if self.labels[i] == label and self.genres[i] != genre]
            same_g_candidates_inds =[i for i in range(self.num_data) if self.labels[i] != label and self.genres[i] == genre]
        elif self.strategy == 'only_same':
            same_l_candidates_inds = [i for i in range(self.num_data) if self.labels[i] == label]
            same_g_candidates_inds = [i for i in range(self.num_data) if self.genres[i] == genre]
        elif self.strategy == 'only_diff':
            same_l_candidates_inds = [i for i in range(self.num_data) if self.genres[i] != genre]
            same_g_candidates_inds = [i for i in range(self.num_data) if self.labels[i] != label]
        elif self.strategy == 'mixture':
            if random.randint(0,1) > 0:
                same_l_candidates_inds = [i for i in range(self.num_data) if self.labels[i] == label]
                same_g_candidates_inds = [i for i in range(self.num_data) if self.genres[i] == genre]
            else:
                same_l_candidates_inds =[i for i in range(self.num_data) if self.labels[i] == label and self.genres[i] != genre]
                same_g_candidates_inds =[i for i in range(self.num_data) if self.labels[i] != label and self.genres[i] == genre]
        elif self.strategy == 'partial':
            if genre < 0:
                same_l_candidates_inds = [i for i in range(self.num_data) if self.labels[i] == label]
                same_g_candidates_inds = [i for i in range(self.num_data) if self.labels[i] != label]
            else:
                same_l_candidates_inds =[i for i in range(self.num_data) if self.labels[i] == label and self.genres[i] != genre]
                same_g_candidates_inds =[i for i in range(self.num_data) if self.labels[i] != label and self.genres[i] == genre]

        if len(same_l_candidates_inds) == 0:
            same_l_candidates_inds = [i for i in range(self.num_data) if self.labels[i] == label]
        if len(same_g_candidates_inds) == 0:
            same_g_candidates_inds = [i for i in range(self.num_data) if self.genres[i] == genre]
        # elif self.strategy == 'bert_distance':
        #     continue
        # elif self.strategy == 'sts_distance':
        #     continue
        # elif self.strategy == 'nli_distance':
        #     continue
        same_l_ind = np.random.choice(same_l_candidates_inds)
        same_g_ind = np.random.choice(same_g_candidates_inds)
        same_l_sample = self.emb_dataset[same_l_ind]
        same_g_sample = self.emb_dataset[same_g_ind]
        return (anchor_data,same_l_sample, same_g_sample)
        
    def __len__(self):
        return len(self.emb_dataset)


def train_TripletBert(args, triplet_train_dataset,triplet_eval_datasets,dev_emb_dataset, model, batch_size, \
    params_to_update = 'fix_bert', max_steps = 2000, num_train_epochs = 20, avg_step = 10, eval_step = 200, transfer_step = 200):
    
    logger = logging.getLogger("root")
    train_sampler = RandomSampler(triplet_train_dataset) if args.local_rank == -1 else DistributedSampler(triplet_train_dataset)

    train_dataloader = DataLoader(triplet_train_dataset, sampler=train_sampler, batch_size=batch_size)
    optimizer_grouped_parameters = [p for n, p in model.named_parameters() ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr = args.learning_rate_triplet)

    # Train!
    logger.info("***** Training TripletBert *****")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch")
    model.to(args.device)
    # flag = 2
    avg_losses = [0.0,0.0,0.0, 0.0]
    train_losses = {}
    val_losses = {}
    transfer_results = {}
    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            
            batch = [[t.to(args.device) for t in trip] for trip in batch]
            outputs = model([batch[0][0], batch[1][0], batch[2][0]])
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            with torch.no_grad():
                avg_losses = [avg_losses[i] + outputs[i].item() for i in range(4)]
            if global_step % avg_step == 0 and global_step > 1:
                logger.warning("TripletBert average training loss %s, loss_sent %s, loss_other %s loss_recon %s at global_step %s", avg_losses[0]/avg_step, avg_losses[1]/avg_step, avg_losses[2]/avg_step,avg_losses[3]/avg_step, global_step)
                train_losses[global_step] = [avg_losses[0]/avg_step, avg_losses[1]/avg_step, avg_losses[2]/avg_step,avg_losses[3]/avg_step]
                avg_losses = [0.0,0.0,0.0,0.0]
            
            if global_step % eval_step == 0 :
                val_losses[global_step] = []
                for triplet_eval_dataset in triplet_eval_datasets:
                    avg_eval_losses= eval_TripletBert(device = args.device, triplet_eval_dataset = triplet_eval_dataset, model = model, batch_size = 2) 
                    logger.warning("TripletBert evaluation loss %s, loss_sent %s, loss_other %s loss_recon %s at global_step %s", avg_eval_losses[0], avg_eval_losses[1], avg_eval_losses[2], avg_eval_losses[3], global_step)
                    val_losses[global_step].append(avg_eval_losses)
            if global_step % transfer_step == 0 :
                #need to ensure the params are not updated at this step
                model_name = "TripletBert_step" + str(global_step) +'.pth'
                torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
                transfer_results[global_step] = lr_eval_DEBert(model, 'TripletBert', triplet_train_dataset, triplet_eval_datasets, args.device)
                logger.warning("TripletBert transfer results  %s at step %s,", transfer_results[global_step], global_step)

            # tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if (max_steps > 0 and global_step > max_steps) :
            train_iterator.close()
            torch.save(model.state_dict(), os.path.join(args.output_dir, "TripletBert.pth"))
            torch.save(args, os.path.join(args.output_dir, "args.pth"))
            break
    
    with open(os.path.join(args.output_dir, "train_losses.json"),"w") as f:
        f.write(json.dumps(train_losses))

    with open(os.path.join(args.output_dir, "val_losses.json"),"w") as f:
        f.write(json.dumps(val_losses))

    with open(os.path.join(args.output_dir, "transfer_results.json"),"w") as f:
        f.write(json.dumps(transfer_results))

    return train_losses, val_losses, transfer_results

def eval_TripletBert(device, triplet_eval_dataset, model, batch_size):
    logger = logging.getLogger('root')
    #
    logger = logging.getLogger("root")
    eval_sampler = RandomSampler(triplet_eval_dataset)
    eval_dataloader = DataLoader(triplet_eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    
    # Train!
    logger.info("***** Evaluating TripletBert *****")

    model.eval()
    model.to(device)
    avg_losses = [0.0] * 4
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = [[t.to(device) for t in trip] for trip in batch]
        with torch.no_grad():
            outputs = model([batch[0][0], batch[1][0], batch[2][0]])
        avg_losses = [avg_losses[i] +outputs[i].item() for i in range(4)]
    avg_losses = [a/len(eval_dataloader) for a in avg_losses]
    logger.info('Evaluation loss is: %s ',avg_losses)
    return avg_losses

# def transfer_TripletBert(args, train_dataset,eval_dataset,orig_eval_dataset, TripletBert_model,transfer_encoder = 'Sentiment',\
#      batch_size = 4, max_steps = 200, num_train_epochs = 10, eval_step = 50 ,params_to_update ='classifier', label_genre = False):
#     final_results = {}
#     model = TransferTripletBert(TripletBert_model, transfer= transfer_encoder)
#     logger = logging.getLogger("root")
#     logger.warning("Transfering %s ", transfer_encoder)
#     train_sampler = RandomSampler(train_dataset)
#     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

#     t_total = min(max_steps,  num_train_epochs *(len(train_dataloader)))
#     eval_step = min(eval_step, t_total//3)

#     if params_to_update == 'classifier':
#         optimizer_grouped_parameters = [p for n, p in model.named_parameters() if 'classifier' in n ]
#     elif params_to_update == 'last_layer_and_classifier':
#         named_params = [(n,p) for n,p in model.named_parameters()]
#         for j in range(11):
#             named_params = [ (n,p) for n,p in named_params if 'layer.'+str(j) not in n ]
#         named_params = [ (n,p) for n,p in named_params if 'embeddings' not in n ]
#         optimizer_grouped_parameters = [p for n, p in named_params]
#     elif params_to_update == 'ALL':
#         optimizer_grouped_parameters = [p for n, p in model.named_parameters() ]
#     optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=5e-5)

#     # Train!
#     logger.info("***** Training TripletBert Classifier*****")

#     global_step = 0
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     train_iterator = trange(int(num_train_epochs), desc="Epoch")
#     model.to(torch.device("cuda:1"))
#     best_acc = 0
#     best_orig_acc = 0
#     for iter in train_iterator:
#         epoch_iterator = tqdm(train_dataloader, desc="Iteration")
#         for step, batch in enumerate(epoch_iterator):
            
#             batch = tuple(t.to(torch.device("cuda:1")) for t in batch)
#             if label_genre:
#                 outputs = model(inputs = batch, transfer = transfer_encoder,labels = batch[4])
#             else:
#                 outputs = model(inputs = batch, transfer = transfer_encoder,labels = batch[3])
#             loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
#             loss.backward()

#             if global_step % eval_step == 0 or global_step == max_steps or (global_step < 100 and global_step % 10 == 0):
#                 result = eval_TransTripletBert(torch.device("cuda:1"), eval_dataset, model, batch_size = 4, label_genre=label_genre)
#                 orig_result = eval_TransTripletBert(torch.device("cuda:1"), orig_eval_dataset, model, batch_size = 4, label_genre=label_genre)
#                 final_results[global_step] = result
#                 final_results[str(global_step) + '_orig'] = orig_result
#                 logger.info("step: %s acc: %s  f1: %s",global_step, result["acc"], result["macro_f1"])
#                 logger.info("Original eval set acc: %s  f1: %s", orig_result["acc"], orig_result["macro_f1"])
#                 if result["acc"] > best_acc:
#                     best_acc = result["acc"]
#                 if orig_result["acc"] > best_orig_acc:
#                     best_orig_acc = orig_result["acc"]
#             tr_loss += loss.item()
#             optimizer.step()
#             model.zero_grad()
#             global_step += 1
#             if max_steps > 0 and global_step > max_steps:
#                 epoch_iterator.close()
#                 break
#         if (max_steps > 0 and global_step > max_steps):
#             train_iterator.close()
            
#             torch.save(args, os.path.join(args.output_dir, "args.pth"))
#             break
#     logger.warning('transfer TripletBert Result for encoder %s', transfer_encoder)
#     logger.warning(final_results)
#     return global_step, tr_loss / global_step, best_acc, best_orig_acc


# def eval_TransTripletBert(device, eval_dataset, model, batch_size, label_genre = False):
#     logger = logging.getLogger("root")
#     eval_sampler = RandomSampler(eval_dataset)
#     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

#     logger.info("***** Evaluating *****")
#     epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
#     preds = None
#     out_label_ids = None
#     for _, batch in enumerate(epoch_iterator):
#         batch = tuple(t.to(device) for t in batch)
        
#         if label_genre:
#             labels = batch[4]
#         else:
#             labels = batch[3]
#         outputs = model(inputs = batch,labels = labels)
#         logits = outputs[1]
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#             out_label_ids = labels.detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#             out_label_ids = np.append(out_label_ids,labels.detach().cpu().numpy(), axis=0)

#     preds = np.argmax(preds, axis=1)
#     simple_accuracy = (preds == out_label_ids).mean()
#     macro_f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
#     result = {"acc": simple_accuracy,
#     "macro_f1": macro_f1}
    
    
#     return result



