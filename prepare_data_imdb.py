import json
import glob
import logging
import os
import random
import matplotlib.pyplot as plt

import numpy as np
import torch

from pytorch_transformers import BertTokenizer

from data_utils import InputExample, InputFeatures,convert_examples_to_features


def get_movie_genre(genre_dir = '/home/xiongyi/dataxyz/data/imdb/title.basics.tsv'):
    tit_2_gen = {}
    with open(genre_dir) as f:
        lines = f.readlines()
    for line in lines[1:]:
        title = line.split('\t')[0]
        genre = line.split('\t') [-1]
        tit_2_gen[title] = genre.strip().split(',')
    return tit_2_gen


def combine_movie_genre(tit_2_gen, movie_dir = '/home/xiongyi/dataxyz/data/imdb/train', visualize = False):
    all_data ={}
    all_genre_counts = {}
    for _, label in enumerate(['neg', 'pos']):
        texts = {}
        text_dir = os.path.join(movie_dir, label)
        text_fns = os.listdir(text_dir)
        for i,fn in enumerate(text_fns):
            if i % 100 == 0:
                print (i, 'in ', len(text_fns))
            # print (fn)
            text_id = int(fn.split('_')[0])
            texts[text_id] = open(os.path.join(text_dir, fn)).readline().strip()
        with open(os.path.join(movie_dir, 'urls_'+label+'.txt')) as f:
            urls = f.readlines()
        urls = [u.strip() for u in urls]
        titles = [ u.split('title/')[1].split('/user')[0] for u in urls]
        genres = [tit_2_gen[t]  if t in tit_2_gen else None  for t in titles]
        all_data[label] = [ (texts[i], genres[i])  for i in range(len(genres))]
        all_genre_counts[label] = {}
        all_genre_counts[label]['None'] = 0
        for g in genres:
            if g is None:
                all_genre_counts[label]['None'] += 1
            else:
                for gg in g:
                    if gg in all_genre_counts[label]:
                        all_genre_counts[label][gg] += 1
                    else:
                        all_genre_counts[label][gg] = 0
    return all_data, all_genre_counts
    '''
    for visualizing genre distribution
    '''
    if visualize:
        all_genre_names = [g for g in list(all_genre_counts['pos'].keys()) if g in list(all_genre_counts['neg'].keys())]
        plt.figure(figsize =(30,15))
        neg_scores = [all_genre_counts['neg'][g] for g in all_genre_names]
        pos_scores = [all_genre_counts['pos'][g] for g in all_genre_names]
        ind = np.arange(len(all_genre_names))    # the x locations for the groups

        p1 = plt.bar(ind, neg_scores)
        p2 = plt.bar(ind, pos_scores,bottom=neg_scores, tick_label= [pos_scores[i]/neg_scores[i] for i in range(len(neg_scores)) ])
        for i in range(len(all_genre_names)):
            pos_rate = "{0:.2f}".format(pos_scores[i]/(pos_scores[i]+neg_scores[i]))
            plt.text(i - 0.25, neg_scores[i] + pos_scores[i], pos_rate, color='k', fontweight='bold')
        plt.ylabel('Num')
        plt.title('Sentiment ratio by genre')
        plt.xticks(ind, all_genre_names)
        plt.legend((p1[0], p2[0]), ('neg', 'pos'))

        plt.savefig('Sentiment_ratio_by_genre')
        plt.show()



def prepare_imdb_subset(all_data, genre_0 = 'Drama', genre_1 = 'Horror', genre_0_pos_ratio = 0.8, sample_size = 2000 ):
    samples = {}
   
    for label_id, label in enumerate(['neg', 'pos']):
        dat = all_data[label]
        dat_0,dat_1 =[], []
        for d in dat:
            if d[-1] is None:
                continue
            if genre_0 in d[-1] and genre_1 not in d[-1]:
                new_d = (d[0], 0)
                dat_0.append(new_d)
            if genre_1 in d[-1] and genre_0 not in d[-1]:
                new_d = (d[0], 1)
                dat_1.append(new_d)
        samples[label] = []
        print (len(dat_0), ' ', len(dat_1))
        if genre_0_pos_ratio > 0: 
            if label == 'pos':
                dat_0_sample_index = np.random.choice(range(len(dat_0)), size = int(genre_0_pos_ratio * sample_size / 2), replace=True)
                dat_1_sample_index = np.random.choice(range(len(dat_1)), size = int((1-genre_0_pos_ratio) * sample_size / 2), replace=True)
            else:
                dat_0_sample_index = np.random.choice(range(len(dat_0)), size = int((1-genre_0_pos_ratio) * sample_size / 2), replace=True)
                dat_1_sample_index = np.random.choice(range(len(dat_1)), size = int(genre_0_pos_ratio * sample_size / 2), replace=True)
            dat_0_samples = [dat_0[i] for i in dat_0_sample_index]
            dat_1_samples = [dat_1[i] for i in dat_1_sample_index]
            samples[label] = dat_0_samples + dat_1_samples
            samples[label] = [s + (label_id,) for s in samples[label]]
        else:
            dat = dat_0 + dat_1
            sample_index = np.random.choice(range(len(dat)) , int(sample_size/2), replace = True)
            samples[label] = [dat[i] for i in sample_index]
            samples[label] = [s + (label_id,) for s in samples[label]]
    return samples['pos'] + samples['neg']


def save_features(data_dir, max_seq_length, line_num, examples, label_list, tokenizer, filename, has_genre = False):
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(
        str(line_num), str(max_seq_length),filename ))
    print ("Creating features from dataset file at %s", data_dir)
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,          
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=1,
        pad_on_left=False,                 
        pad_token_segment_id=0, has_genre = has_genre)
    print ("Saving features into cached file %s", cached_features_file)
    print (set([f.label_id for f in features]))
    print (set([f.genre_id for f in features]))
    torch.save(features, cached_features_file)



if __name__ == "__main__":
    print ('Start')
    GENRE_0 = 'Drama'
    GENRE_1 = 'Horror'
    tit_2_gen = get_movie_genre(genre_dir = '/home/xiongyi/dataxyz/data/imdb/title.basics.tsv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    for prefix in ['train','test']:
        movie_dir = os.path.join('/home/xiongyi/dataxyz/data/imdb', prefix)
        all_data, all_genre_counts = combine_movie_genre(tit_2_gen, movie_dir=movie_dir)
        
        for genre_0_pos_ratio in [0.05,0.15, 0.25,0.5,0.75,0.85,0.95,-1]:
            print ('*#$' * 10)
            print (genre_0_pos_ratio )
            if prefix == 'train':
                sample_size = 3000
            elif prefix == 'test':
                sample_size = 400
            samples = prepare_imdb_subset(all_data, genre_0=GENRE_0, genre_1 = GENRE_1, genre_0_pos_ratio = genre_0_pos_ratio, sample_size = sample_size)
            examples = []
            for i,s in enumerate(samples):
                examples.append(InputExample(i,s[0], None, s[2], s[1]))
            feature_dir = os.path.join('/home/xiongyi/dataxyz/data/imdb/','features', GENRE_0+'_'+GENRE_1)
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            save_features(data_dir=feature_dir, max_seq_length=512,\
                 line_num=None,examples=examples, label_list=[0,1],\
                tokenizer=tokenizer, filename=prefix +'_ratio_'+str(genre_0_pos_ratio)+'_size_'+str(sample_size),\
                has_genre= True)
            #Also save the raw sentences
            raw_sents = [s[0] for s in samples]
            raw_sents_filename = 'raw_sents_'+prefix +'_ratio_'+str(genre_0_pos_ratio)+'_size_'+str(sample_size)
            with open(os.path.join(feature_dir,raw_sents_filename), 'w') as f:
                for s in raw_sents:
                    f.write(s+'\n')