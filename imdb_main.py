import argparse
import glob
import logging
import os, json
import random
import datetime
import gc
from distutils.dir_util import copy_tree

from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,AdamW, WarmupLinearSchedule)

from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from tensorboardX import SummaryWriter

from data_utils import InputExample, InputFeatures,load_dataset
from util_funcs import train, evaluate, lr_eval_Bert,  MTBert, calc_emb
from plot_utils import plot_results_DEBert
from Adv_Bert import AdvBert, train_AdvBert
from TripletBert import TripletBert, TripletDataset, train_TripletBert

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


#TODO write several scripts for: preparing the data, finetuning bert, training TripletBert, doing both, making plots?
def main(margin = 10, shuffle = False, train_Bert = False, random_neg = False, output_dir = None, model_dir = None,\
    fine_tune_option = 'finetune_sent', pretrained_path = None,strategy = 'opposite', rand_init =False,
    beta_recon = 0, learning_rate_triplet = 1e-4,learning_rate_bert = 5e-5, candidate_num = 20, hs = 300,method = 'factor', aux_label_ratio = 1,
    train_ratio = 0.5, beta_dis = 1, pred_both = False, multi_task = False, num_labeled_data = 500, train_dataset = None, dev_dataset = None):
    
    assert (pred_both and multi_task) is False,  " MT and PB can't both be true!"
    # wandb.init(project="dbert") 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="/home/xiongyi/dataxyz/data/imdb/features/Drama_Horror", type=str, required=False,
                        help="The input data dir for the cached pre-processed data file")

    parser.add_argument("--model_type", default="bert", type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    
    parser.add_argument("--model_name_or_path", default="/home/xiongyi/dataxyz/repos/DBert/models/IMDB_orig_bert", type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="Path to store checkpoints: ")
    
    ## Other parameters that we care about
    
    parser.add_argument("--num_of_trials", default=5, type=int,
                        help="Path to store checkpoints: ")
    parser.add_argument("--margin", default=1.0, type=float,
                        help="Margin for triplet loss: ")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")

    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--early_stop", default=-1, type=int,
                        help="If > 0: stop if eval_f1 does not increase after this many logging_steps")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument('--transfer_params', default = "fix_bert", type = str,
                        help = "Choose between ALL, LAST")

    parser.add_argument("--shuffle_baseline", action='store_true',
                        help="Whether to use a random baseline.")
    ## Parameters that we don't really care about

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--output_mode", default="classification", type=str,
                        help="classification of regression")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true', default= True,
                        help="Rul evaluation during training at each logging step.")
    
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate_bert", default=5e-5, type=float,    
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")


    parser.add_argument("--warmup_steps", default=50, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    
    if output_dir is None:
        output_dir = datetime.datetime.now().strftime('%m%d%H%M%S') + 'movie_transfer'
    args = parser.parse_args(["--output_dir", output_dir, "--do_train", "--do_lower_case",
          "--overwrite_output_dir", "--logging_steps","50", "--save_steps","50", "--max_steps", "300","--margin", str(margin)])

          
    args.shuffle_baseline = shuffle
    args.learning_rate_bert = learning_rate_bert
    args.learning_rate_triplet = learning_rate_triplet

    args.candidate_num = candidate_num
    # output dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = logging.getLogger("root")
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.info)

    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    fh = logging.FileHandler(os.path.join(args.output_dir, 'log.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.warning('Margins are both %s', margin )
    if shuffle:
        logger.warning('Shuffled!')
    # Set seed

    # Prepate data
    if pred_both:
        num_labels = 4
    else:
        num_labels = 2
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else 'bert-base-uncased', num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    
    
    logger.warning("Training/evaluation parameters %s", args)
    if train_dataset is None:
        num_train = num_labeled_data
        train_dataset = load_dataset(data_dir = args.data_dir, num_data = num_train, file_name= 'train_ratio_'+str(train_ratio)+'_size_3000',load_genre= True)
    
    eval_ratios = [0.05, 0.25, 0.5, 0.75, 0.95]

    eval_datasets = []
    for ratio in eval_ratios:
        eval_dataset = load_dataset(data_dir = args.data_dir, num_data = 400, \
            file_name= 'test_ratio_'+str(ratio)+'_size_400',load_genre= True)
        eval_datasets.append(eval_dataset)    # orig_eval_dataset = None
    #TODO: load validation data


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # Train original BERT
    if train_Bert:
        #args.evaluate_during_training = False
        if rand_init:
            config.num_hidden_layers = 4
            config.num_attention_heads = 6
            model = model_class(config)
        else:
            model = model_class.from_pretrained('bert-base-uncased', from_tf=False, config=config)

        if multi_task:
            model = MTBert(model)
        # Distributed and parallel training
        model.to(args.device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        #TODO Add early stopping
        train(args = args, train_dataset = train_dataset, model = model, pred_both = pred_both, multi_task= multi_task, \
            multiple_eval_datasets= eval_datasets, dev_dataset = dev_dataset)

        if multi_task:
            model_to_save = model.module.bert if hasattr(model, 'module') else model.bert 
        else:
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(model_dir)
        torch.save(args, os.path.join(model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", model_dir)
    else:
        model = model_class.from_pretrained(model_dir, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        # model = model_class.from_pretrained('bert-base-uncased')
        model.to(args.device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
    ###Disentangle the trained model
    args.evaluate_during_training = True


    if method == 'TripletBert':    
        ##construct triplets 
        #TODO construct triplet validation set
        train_embs = calc_emb(dataset= train_dataset, model= model, device=args.device, model_type = 'Bert')
        train_emb_dataset = TensorDataset(torch.Tensor(train_embs), torch.LongTensor([d[-3] for d in train_dataset])\
            ,torch.LongTensor([d[-2] for d in train_dataset],))

        dev_embs = calc_emb(dataset= dev_dataset, model= model, device=args.device, model_type = 'Bert')
        dev_emb_dataset = TensorDataset(torch.Tensor(dev_embs), torch.LongTensor([d[-3] for d in dev_emb_dataset])\
            ,torch.LongTensor([d[-2] for d in dev_emb_dataset],))
        



        num_aux_labels = int(aux_label_ratio * len(train_embs))

        if strategy == 'opposite':
            #if strategy is opposite we remove all the data without genre labels.
            pos_inds = [i for i in range(len(train_dataset)) if train_dataset[i][-3] == 1 ]
            neg_inds = [i for i in range(len(train_dataset)) if train_dataset[i][-3] == 0 ]
            sub_inds = pos_inds[:int(num_aux_labels/2)] + neg_inds[:int(num_aux_labels/2)]
            train_emb_dataset = [train_emb_dataset[i] for i in sub_inds]
        elif strategy == 'partial':
            pass
        
        triplet_train_dataset = TripletDataset(train_emb_dataset, strategy=strategy) 
        triplet_eval_datasets = []
        for ratio in eval_ratios:
            eval_dataset = load_dataset(data_dir = args.data_dir, num_data = 400, \
                file_name= 'test_ratio_'+str(ratio)+'_size_400',load_genre= True)
            eval_embs = calc_emb(dataset= eval_dataset, model= model, device=args.device, model_type = 'Bert')
            eval_emb_datasets = TensorDataset(torch.Tensor(eval_embs), torch.LongTensor([d[-3] for d in eval_dataset])\
            ,torch.LongTensor([d[-2] for d in eval_dataset],))
            triplet_eval_datasets.append(TripletDataset(eval_emb_datasets, random_neg = random_neg))
        
        #TODO: add early stopping
        TripletBert_model = TripletBert(margin_sent= args.margin, margin_other= args.margin, beta_recon= beta_recon, hidden_size=hs, output_size=hs)

        train_losses, val_losses, transfer_results = train_TripletBert(args = args, triplet_train_dataset=triplet_train_dataset, triplet_eval_datasets=triplet_eval_datasets, dev_emb_dataset=dev_emb_dataset, model=TripletBert_model, \
            batch_size = 16, params_to_update='fix_bert', max_steps=300, num_train_epochs=10, avg_step=10, eval_step=30, transfer_step=30)
    # elif method == 'adv':
    #     ##calculate embeddings
    #     train_embs = calc_emb_BertFS(dataset= train_dataset, model= model, device=args.device)
    #     all_eval_emb_datasets = []
    #     for ratio in eval_ratios:
    #         eval_dataset = load_dataset(data_dir = args.data_dir, num_data = 100, \
    #             file_name= 'test_ratio_'+str(ratio)+'_size_400',load_genre= True)
    #         eval_embs = calc_emb_BertFS(dataset= eval_dataset, model= model, device=args.device)
    #         eval_emb_datasets = TensorDataset(torch.Tensor(eval_embs), torch.LongTensor([d[-3] for d in eval_dataset])\
    #         ,torch.LongTensor([d[-2] for d in eval_dataset],))
    #         all_eval_emb_datasets.append(eval_emb_datasets)
        
    #     train_emb_dataset = TensorDataset(torch.Tensor(train_embs), torch.LongTensor([d[-3] for d in train_dataset])\
    #         ,torch.LongTensor([d[-2] for d in train_dataset],))

        # if method == 'factor':
        #     FB_model = FactorBert(input_size=train_embs.shape[1])
        #     train_losses, val_losses, transfer_accs,weight_ratios = train_FactorBert(args, train_emb_dataset, eval_emb_dataset, \
        #         orig_eval_emb_dataset, FB_model, batch_size = 32, max_steps = 2000,avg_step = 10, eval_step = 200, transfer_step = 200)
        # # elif method == 'adv':
        # model = AdvBert(input_size=train_embs.shape[1], beta_dis = beta_dis)
        # train_losses, val_losses, transfer_results  = train_AdvBert(args, train_emb_dataset, all_eval_emb_datasets, \
        #     model, batch_size = 32, max_steps = 200,avg_step = 10, eval_step = 20, transfer_step = 20)
    elif method == 'finetune_only':
        return

    # train_losses = json.load(open(os.path.join(args.output_dir, 'train_losses.json')))
    # val_losses = json.load(open(os.path.join(args.output_dir, 'val_losses.json')))
    # transfer_results = json.load(open(os.path.join(args.output_dir,'transfer_results.json')))
    # plot_results_DEBert( train_losses, val_losses, transfer_results,args.output_dir, model_type = 'TripletBert',ratio_ids = [0,1,5,6,8,9])
    
    

if __name__ == "__main__":
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    for rep in [0,1,2,3,4]:
        root_res_dir = '/home/xiongyi/dataxyz/experiments/Disentangle_Imdb/2_6_new/rep_{}'.format(rep)
        if not os.path.exists(root_res_dir):
            os.makedirs(root_res_dir)
        for train_ratio in [0.05,0.15, 0.25, 0.5,]:
            full_train_data = load_dataset(data_dir = "/home/xiongyi/dataxyz/data/imdb/features/Drama_Horror", num_data = 3000, file_name= 'train_ratio_'+str(train_ratio)+'_size_3000',load_genre= True, shuffle=False)
            method = 'TripletBert'
            dim = 300
            if os.path.exists( os.path.join(root_res_dir, 'shuffled_pos_inds.txt') ):
                assert os.path.exists( os.path.join(root_res_dir, 'shuffled_neg_inds.txt'))

                with open(os.path.join(root_res_dir, 'shuffled_pos_inds.txt'), 'r') as f:
                    pos_inds = f.readline().strip().split(',')
                pos_inds = [int(i) for i in pos_inds]
                with open(os.path.join(root_res_dir, 'shuffled_neg_inds.txt'), 'r') as f:
                    neg_inds = f.readline().strip().split(',')
                neg_inds = [int(i) for i in neg_inds]
            else:
                print ('loading pre-defined inds')
                pos_inds = [i for i in range(len(full_train_data)) if full_train_data[i][-3].item() == 1]
                neg_inds = [i for i in range(len(full_train_data)) if full_train_data[i][-3].item() == 0]
               
                # first load the full dataset and shuffle it, so that when we increase the size of the training set it would be a superset of the previous one
                random.shuffle(pos_inds)
                random.shuffle(neg_inds)
                with open(os.path.join(root_res_dir, 'shuffled_pos_inds.txt'), 'w') as f:
                    f.write(','.join([str(i) for i in pos_inds]))
                with open(os.path.join(root_res_dir, 'shuffled_neg_inds.txt'), 'w') as f:
                    f.write(','.join([str(i) for i in neg_inds]))

           
            for num_labeled_data in [250,]:
                num_train = int(0.8*num_labeled_data)
                train_idx = np.random.choice(num_labeled_data, num_train, replace = False)
                dev_idx = [i for i in range(num_labeled_data) if i not in train_idx]
                small_pos_indices = [ pos_inds[i] for i in train_idx]
                small_neg_indices = [ neg_inds[i] for i in train_idx]
                dev_inds = [ pos_inds[i] for i in dev_idx] + [ neg_inds[i] for i in dev_idx]
                dev_dataset =  [full_train_data[i] for i in dev_inds ]
                for aux_label_ratio in [0.01,0.1,0.2,0.5,0.8,1]:
                    num_aux_labels = int(num_train * aux_label_ratio)
                    # We should fix the training set at this point
                    no_genre_indices = pos_inds[num_aux_labels:num_train] + neg_inds[num_aux_labels:num_train]
                    small_indices = small_pos_indices + small_neg_indices
                    small_train_dataset = [full_train_data[i] for i in small_indices ]
                    for i in small_indices:
                        if i in no_genre_indices:
                            small_train_dataset.append((full_train_data[i][0],full_train_data[i][1],full_train_data[i][2], full_train_data[i][3], torch.tensor(-1),full_train_data[i][5]))
                        else:
                            small_train_dataset.append(full_train_data[i])


                    model_dir_name_fmt = 'finetuned_bert_train_ratio_{}_{}_num_labeled_{}_aux_ratio_{}'
                    model_dir_both = os.path.join(root_res_dir, model_dir_name_fmt.format(train_ratio,'pred_both',num_labeled_data, aux_label_ratio) )
                    model_dir_sent = os.path.join(root_res_dir, model_dir_name_fmt.format(train_ratio,'pred_sent',num_labeled_data, 0) )
                    # model_dir_sent_init = os.path.join(root_res_dir, model_dir_name_fmt.format(train_ratio,'pred_sent_init',num_labeled_data) )
                    # model_dir_null = os.path.join(root_res_dir, model_dir_name_fmt.format(train_ratio,'for_comparison',num_labeled_data) )

                    finetune_output_dir_fmt = 'finetune_result_train_ratio_{}_{}_num_labeled_{}_aux_ratio_{}'
                    finetune_output_dir_both = os.path.join(root_res_dir, finetune_output_dir_fmt.format(train_ratio,'pred_both',num_labeled_data, aux_label_ratio) )
                    finetune_output_dir_sent = os.path.join(root_res_dir, finetune_output_dir_fmt.format(train_ratio,'pred_sent',num_labeled_data, 0) )
                    # finetune_output_dir_sent_init = os.path.join(root_res_dir, finetune_output_dir_fmt.format(train_ratio,'pred_sent_init',num_labeled_data) )
                    # finetune_output_dir_null = os.path.join(root_res_dir, finetune_output_dir_fmt.format(train_ratio,'for_comparison',num_labeled_data) )

                    # #first finetune the bert model and save it
                    # main(train_Bert= True, method = 'finetune_only', train_dataset=small_train_dataset, dev_dataset = None, multi_task= False, output_dir=finetune_output_dir_sent, model_dir = model_dir_sent, num_labeled_data = num_labeled_data)
                    main(train_Bert= True, method = 'finetune_only', train_dataset=small_train_dataset, dev_dataset = dev_dataset, multi_task = True, output_dir=finetune_output_dir_both, model_dir = model_dir_both, num_labeled_data = num_labeled_data)
                    # main(train_Bert= True, method = 'finetune_only', train_dataset=small_train_dataset, multi_task = False, output_dir=finetune_output_dir_sent_init, model_dir = model_dir_both, num_labeled_data = num_labeled_data,learning_rate_bert = 1e-4, rand_init = True)
                    
                    for strategy in ['opposite',]:
                            # if aux_label_ratio == 0 and strategy == 'opposite':
                                # continue
                            outdir_name_both = os.path.join(root_res_dir,  method +'_'+ str(train_ratio) +'_'+strategy +'_pred_both' +'_aux_ratio_'+str(aux_label_ratio) +'_num_labeled_'+str(num_labeled_data) )
                            main(train_Bert= False, method = method,train_dataset=small_train_dataset, multi_task = True, output_dir=outdir_name_both,model_dir = model_dir_both, num_labeled_data = num_labeled_data, strategy=strategy)
                            logger.warning('finisehd one model')
                            # outdir_name_sent = os.path.join(root_res_dir,  method +'_'+ str(train_ratio) +'_'+strategy +'_pred_sent' +'_aux_ratio_'+str(aux_label_ratio) +'_inner_rep_' +str(inner_rep) +'_num_labeled_'+str(num_labeled_data) )
                            # torch.cuda.empty_cache() 
                            # main(train_Bert= False, method = method,train_dataset=small_train_dataset, output_dir=outdir_name_sent,model_dir = model_dir_sent,num_labeled_data = num_labeled_data, aux_label_ratio =aux_label_ratio, strategy=strategy )
                            # logger.warning('finisehd one model')
                            # torch.cuda.empty_cache() 
