import torch.nn as nn
import numpy as np
from itertools import combinations
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from torch.nn import CrossEntropyLoss

import os,logging,argparse, datetime,json

import glob
import random

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from torch.utils.data import Dataset
from data_utils import InputExample, InputFeatures,load_dataset
from util_funcs import train, evaluate, calc_emb, lr_eval_Bert, lr_eval_DEBert
# from shutil import copyfile, copytree
from distutils.dir_util import copy_tree


from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import matthews_corrcoef, f1_score


"""
Disentangle a pre-trained model using either triplet loss or some VAE or a combination of the two.
Takes the original model as input, return the disentangled new model.
"""


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
class AdvBert(nn.Module):
    def __init__(self,input_size = 768, hidden_size = 300, cls_hs = 200, dis_hs = 200, output_size = 300,\
         beta_other = 1, beta_dis= 1, beta_dis_other = 1):
        super(AdvBert, self).__init__()
        self.output_size = output_size
        self.common_encoder = nn.Sequential(nn.Linear(input_size, hidden_size),\
             nn.ReLU(), )
        self.encoder_sent = nn.Sequential( nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(), nn.Linear(hidden_size, output_size),nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
              nn.ReLU(), nn.Linear(hidden_size, output_size),nn.ReLU())
        self.encoder_other = nn.Sequential(nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(), nn.Linear(hidden_size, output_size),nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
              nn.ReLU(), nn.Linear(hidden_size, output_size),nn.ReLU())
        
        self.cls_sent = Classifier(emb_dim = output_size, hidden_size = cls_hs)
        self.cls_other = Classifier(emb_dim = output_size, hidden_size = cls_hs)
        self.dis_sent = Discriminator(emb_dim = output_size, hidden_size = dis_hs)
        self.dis_other = Discriminator(emb_dim = output_size, hidden_size = dis_hs)
        
        self.beta_other = beta_other
        self.beta_dis = beta_dis
        self.beta_dis_other = beta_dis_other


    def forward(self, input_embeddings, labels_sent = None, labels_other = None):
        batch_size = input_embeddings.shape[0]
        common_embs =  self.common_encoder(input_embeddings)
        sent_output = self.encoder_sent(common_embs)
        other_output = self.encoder_other(common_embs)


        loss_sent = self.cls_sent(sent_output, labels = labels_sent)[-1]      
        loss_other = self.cls_sent(other_output, labels = labels_other)[-1]      
        disc_loss_sent = self.dis_other(sent_output, labels = labels_other)[-1]
        disc_loss_other = self.dis_sent(other_output, labels = labels_sent)[-1]
        return loss_sent, loss_other,disc_loss_sent,disc_loss_other


class Classifier(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(), nn.Linear(hidden_size, 2))
    def forward(self, embs, labels = None):
        logits = torch.sigmoid(self.mlp(embs))
        if labels is not None:
            loss_fn = torch.nn.NLLLoss()
            loss = loss_fn(logits, labels)
            return logits, loss
        return logits


class Discriminator(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(),nn.Linear(hidden_size, hidden_size),\
             nn.ReLU(), nn.Linear(hidden_size, 2))
    def forward(self, embs, labels = None):
        logits = torch.sigmoid(self.mlp(embs))
        if labels is not None:
            loss_fn = torch.nn.NLLLoss()
            loss = loss_fn(logits, labels)
            return logits, loss
        return logits


def train_AdvBert(args, train_emb_dataset, all_eval_emb_datasets, AB_model, \
    batch_size, max_steps = 2000, num_train_epochs = 100, avg_step = 10, eval_step = 200, transfer_step = 200):

    logger = logging.getLogger("root")
    train_sampler = RandomSampler(train_emb_dataset) if args.local_rank == -1 else DistributedSampler(train_emb_dataset)
    train_dataloader = DataLoader(train_emb_dataset, sampler=train_sampler, batch_size=batch_size)
    
    
    optimizer_g_params = [p for n, p in AB_model.named_parameters() if "dis" not in n ]
    optimizer_d_params = [p for n, p in AB_model.named_parameters() if "dis" in n ]

    optimizer_g = torch.optim.Adam(optimizer_g_params, lr = args.learning_rate)
    optimizer_d = torch.optim.Adam(optimizer_d_params, lr = args.learning_rate)

    # Train!
    logger.info("***** Training DBert *****")

    global_step = 0
    AB_model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch")
    AB_model.to(args.device)
    avg_losses = None
    train_losses, val_losses, transfer_results  = {},{},{}
    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            loss_sent, loss_other,disc_loss_sent,disc_loss_other\
                 = AB_model(input_embeddings= batch[0], labels_sent = batch[1], labels_other = batch[2])
            loss_D = -disc_loss_sent - AB_model.beta_dis_other * disc_loss_other
            # loss_G =  loss_sent + AB_model.beta_other * loss_other - AB_model.beta_dis * loss_D
            loss_G = -loss_D
            with torch.no_grad():
                if avg_losses is None:
                    avg_losses = np.asanyarray([l.item() for l in [loss_sent, loss_other,disc_loss_sent,disc_loss_other] ])
                else:
                    avg_losses += np.asanyarray([l.item() for l in [loss_sent, loss_other,disc_loss_sent,disc_loss_other] ])
            if global_step % avg_step == 0 and global_step > 1:
                logger.warning("FBert average loss_sent , loss_other , tc_loss, disc_loss %s at global_step %s",\
                    list(avg_losses[:3] / avg_step), global_step)
                train_losses[global_step] = list(avg_losses / avg_step)
                avg_losses = np.asanyarray([0.0 for _ in avg_losses])
            
            if global_step % eval_step == 0 :
                val_losses[global_step] = []
                for eval_emb_dataset in all_eval_emb_datasets: 
                    avg_eval_losses= eval_AdvBert(device = args.device, emb_dataset = eval_emb_dataset, model = AB_model, batch_size = 64) 
                    # logger.warning("FBert evaluation loss %s, at global_step %s", avg_eval_losses, global_step)
                    val_losses[global_step].append(avg_eval_losses)
            if global_step % transfer_step == 0 :
                #need to ensure the params are not updated at this step
                model_name = "FBert_step" + str(global_step) +'.pth'
                torch.save(AB_model.state_dict(), os.path.join(args.output_dir, model_name))

                transfer_results[global_step] = lr_eval_DEBert(args, AB_model, 'AdvBert', train_emb_dataset, all_eval_emb_datasets, args.device)
                
                '''
                transfer_accs[global_step] = {}
                transfer_accs[global_step] = {}
                transfer_accs[global_step]['Sent'] = output_sent_pred[0]
                transfer_accs[global_step]['Sent_orig'] = output_sent_pred_orig[0]
                transfer_accs[global_step]['Genre'] = output_genre_pred[0]
                transfer_accs[global_step]['Genre_orig'] = output_genre_pred_orig[0]
                
                weight_ratios[global_step] = {}
                weight_ratios[global_step]['Sent'] = output_sent_pred[1]
                weight_ratios[global_step]['Sent_orig'] = output_sent_pred_orig[1]
                weight_ratios[global_step]['Genre'] = output_genre_pred[1]
                weight_ratios[global_step]['Genre_orig'] = output_genre_pred_orig[1]
                '''
                # AB_model.to(args.device)
            # tr_loss += loss.item()
            if (step + 1) % 5== 0:
                loss_D.backward(retain_graph = True)
                optimizer_d.step()
                AB_model.zero_grad()
            if (step + 1) % 1 == 0:
                loss_G.backward(retain_graph = True)
                optimizer_g.step()
                AB_model.zero_grad()
                global_step += 1
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if (max_steps > 0 and global_step > max_steps) :
            train_iterator.close()
            torch.save(AB_model.state_dict(), os.path.join(args.output_dir, "FBert_final.pth"))
            torch.save(args, os.path.join(args.output_dir, "args.pth"))
            break
    
    with open(os.path.join(args.output_dir, "train_losses.json"),"w") as f:
        f.write(json.dumps(train_losses))

    with open(os.path.join(args.output_dir, "val_losses.json"),"w") as f:
        f.write(json.dumps(val_losses))

    with open(os.path.join(args.output_dir, "transfer_results.json"),"w") as f:
        f.write(json.dumps(transfer_results))

    # with open(os.path.join(args.output_dir, "weight_ratios.json"),"w") as f:
    #     f.write(json.dumps(weight_ratios))

    return train_losses, val_losses, transfer_results 

def eval_AdvBert(device, emb_dataset, model, batch_size):
    logger = logging.getLogger('root')
    #    logger = logging.getLogger("root")
    eval_sampler = RandomSampler(emb_dataset)
    eval_dataloader = DataLoader(emb_dataset, sampler=eval_sampler, batch_size=batch_size)
    
    # Train!
    logger.info("***** Evaluating DBert *****")
    model.eval()
    model.to(device)
    avg_losses = [0.0] * 4
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(input_embeddings= batch[0], labels_sent = batch[1], labels_other = batch[2])
        avg_losses = [avg_losses[i] +outputs[i].item() for i in range(4)]
    avg_losses = [a/len(eval_dataloader) for a in avg_losses]
    logger.info('Evaluation loss is: %s ',avg_losses)
    return avg_losses



def calc_emb_AdvBert(dataset, model, device):
    all_embs = [None for _ in range(3)]
    train_dataloader = DataLoader(dataset, sampler=None, batch_size=8)
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            #logits = self.classifier(cls_embs)
            common_embs =  model.common_encoder(batch[0])
            # sent_embs = model.encoder_sent(common_embs)
            # other_embs = model.encoder_other(common_embs)
            dim = int(common_embs.shape[1]/2)
            sent_embs =common_embs[:, :dim]
            other_embs =common_embs[:, dim:]

            combined_embs = torch.cat((sent_embs,other_embs), dim=1)
            for i,embs in enumerate([sent_embs, other_embs, combined_embs]):
                if all_embs[i] is None:
                    all_embs[i] = embs.detach().cpu().numpy()
                else:
                    all_embs[i] = np.concatenate((all_embs[i], embs.detach().cpu().numpy()), axis=0)
    return all_embs

    
def lr_AdvBert(args, train_emb_dataset, eval_emb_dataset, AB_model, label_type = 'sent'):
    all_train_embs = calc_emb_AdvBert(train_emb_dataset, AB_model, args.device)
    all_eval_embs = calc_emb_AdvBert(eval_emb_dataset, AB_model, args.device)
    all_accs = []
    all_weights = []
    for i,encoder_name in enumerate(['sent_enc', 'other_enc', 'combined']):
        train_embs = all_train_embs[i]
        normalizer = Normalizer()
        train_embs = normalizer.fit_transform(train_embs) 

        eval_embs= all_eval_embs[i]
        eval_embs = normalizer.transform(eval_embs)
        if label_type == 'sent':
            train_labels = [d[-2] for d in train_emb_dataset]
            eval_labels = [d[-2] for d in eval_emb_dataset]
        elif label_type == 'genre':
            train_labels = [d[-1] for d in train_emb_dataset]
            eval_labels = [d[-1] for d in eval_emb_dataset]
            
        lr_model = LogisticRegression(random_state=0, penalty='l2', solver = 'liblinear')
        
        lr_model.fit(X = train_embs, y = train_labels)
        y_pred = lr_model.predict(eval_embs)
        acc = sum(y_pred == eval_labels)/len(y_pred)
        if encoder_name == 'combined':
            weights = lr_model.coef_[0]
            dim = int(len(weights) / 2)
            weght_ratio = np.linalg.norm(weights[:dim])/ np.linalg.norm(weights[dim:])
        all_accs.append(acc)

    return all_accs, weght_ratio


def eval_TransDBert(device, eval_dataset, model, batch_size):
    logger = logging.getLogger("root")
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    logger.info("***** Evaluating *****")
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    preds = None
    out_label_ids = None
    for _, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        outputs = model(inputs = batch,labels = batch[3])
        logits = outputs[1]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    simple_accuracy = (preds == out_label_ids).mean()
    macro_f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    result = {"acc": simple_accuracy,
    "macro_f1": macro_f1}
    
    return result




