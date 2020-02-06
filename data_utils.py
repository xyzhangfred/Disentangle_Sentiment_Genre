import os
import logging, random

import numpy as np
import torch
from torch.utils.data import TensorDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, genre = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.genre = genre

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, genre_id = None,aux_embedding = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.genre_id = genre_id
        self.aux_embedding = aux_embedding

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, has_genre = False,auxiliary_model = None):

    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    #TODO: Modify this so that we also precompute the (Bert/STS/NLI) embeddings if needed.

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            # if len(tokens_a) > max_seq_length - 2:
            #     print (example.text_a, len(tokens_a), ' Too Long')
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        
        if has_genre:
            genre_id = example.genre
        aux_embedding = None
        if auxiliary_model is not None:
            #TODO: embed using the auxiliary_model 
            aux_embedding = auxiliary_model(example.text_a)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id, genre_id=genre_id, aux_embedding=aux_embedding))
    return features

def load_dataset(data_dir, num_data ,file_name, balance = True, shuffle = True,\
     load_task_id = False, max_sequence_length = 512, load_genre = False, load_sent_id = False):
    logger = logging.getLogger("root")
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(None, str(max_sequence_length),file_name))

    logger.info("Loading features from cached file %s", cached_features_file)
    
    features = torch.load(cached_features_file)
    for i,f in enumerate(features):
        f.u_id = i
    # all_label_id = [f.label_id for f in features]
    #print ("label_set: ", label_set)
    total_num = len(features)
    NUM_CATEGORY = 2 #Assuming that we have 2 categories here!
    if balance:
        features_by_lid = [[f for f in features if f.label_id == i] for i in range(NUM_CATEGORY)]
        all_chosen_features = []
        
        if balance == True and num_data is not None:
            num_per_cat = num_data // NUM_CATEGORY  
            for i in range(NUM_CATEGORY):     
                chosen_features = np.random.choice(features_by_lid[i], size = num_per_cat, replace= True)
                all_chosen_features += list(chosen_features)
    elif num_data is not None:
        rand_idx = np.random.randint(0,total_num, num_data)
        all_chosen_features = [features[i] for i in rand_idx]
    else:
        all_chosen_features = features
    if shuffle:
        random.shuffle(all_chosen_features)
    #features = features[:int(0.001*len(features))] 
    all_input_ids = torch.tensor([f.input_ids for f in all_chosen_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in all_chosen_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in all_chosen_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in all_chosen_features], dtype=torch.long)
    all_sent_ids = torch.tensor([f.u_id for f in all_chosen_features], dtype=torch.long)

    if load_task_id:
        all_task_ids = torch.tensor([f.task_id for f in all_chosen_features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_task_ids,all_sent_ids)
    elif load_genre:
        all_genre_ids = torch.tensor([f.genre_id for f in all_chosen_features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_genre_ids,all_sent_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_sent_ids)
    
    return dataset

def load_raw_dataset(data_dir, num_data ,file_name, balance = True,shuffle = False,\
     load_task_id = False, max_sequence_length = 512, load_genre = False):
    '''
    load data in the form of [sentence, senti_label, genre_label]
    '''

    logger = logging.getLogger("root")
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(None, str(max_sequence_length),file_name))

    logger.info("Loading features from cached file %s", cached_features_file)
    
    features = torch.load(cached_features_file)
    all_label_id = [f.label_id for f in features]
    label_set = list(set(all_label_id))
    #print ("label_set: ", label_set)
    total_num = len(features)
    
    if balance:
        features_by_lid = [[f for f in features if f.label_id == i] for i in range(2)]
        all_chosen_features = []
        
        if balance == True and num_data is not None:
            num_per_cat = num_data // 2
            for i in range(2):
                chosen_features = np.random.choice(features_by_lid[i], size = num_per_cat, replace= True)
                all_chosen_features += list(chosen_features)
    elif num_data is not None:
        rand_idx = np.random.randint(0,total_num, num_data)
        all_chosen_features = [features[i] for i in rand_idx]
    else:
        all_chosen_features = features
    if shuffle:
        random.shuffle(all_chosen_features)
    #features = features[:int(0.001*len(features))] 
    all_input_ids = torch.tensor([f.input_ids for f in all_chosen_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in all_chosen_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in all_chosen_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in all_chosen_features], dtype=torch.long)
    if load_task_id:
        all_task_ids = torch.tensor([f.task_id for f in all_chosen_features], dtype=torch.long)
    if load_genre:
        all_genre_ids = torch.tensor([f.genre_id for f in all_chosen_features], dtype=torch.long)

    if load_task_id:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_task_ids)
    elif load_genre:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_genre_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return dataset
