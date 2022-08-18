import numpy as np
import json
import os
from collections import defaultdict, namedtuple
import random
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class ABSATokenizer(BertTokenizer):
    '''
    The text should have been pre-processed before, only do sub word tokenizer here.
    '''
    def subword_tokenize(self, tokens, labels):  # for AE
        split_tokens, split_labels = [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token) #if not token.startswith('##') else [token]
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)

                if labels[ix].startswith('B') and jx != 0:
                    split_labels.append(labels[ix].replace('B', 'I'))
                else:
                    split_labels.append(labels[ix])

                idx_map.append(ix)
        return split_tokens, split_labels, idx_map



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask=output_mask


class PairInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a, input_ids_b, input_mask, segment_ids, label_id):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)
# 892:1***Boot time is super fast , around anywhere from 35 seconds to 1 minute .***B I O O O O O O O O O O O O O***NN NN VBZ JJ RB , IN RB IN CD NNS TO CD NN .

    def read_txt(self, input_file,times=1):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for time in range(times):
            for _, t in enumerate(text):
                if 'raw_data' in input_file:
                    t = t.split('####')[1]  # 'This=O is=O a=O example=O'
                    sentence = [i.split('=')[0].lower() for i in t.split() if len(i.split('=')[0])]
                    label = [i.split('=')[1] for i in t.split() if len(i.split('=')[0])]
                else:
                    sentence, label = t.split('####')  # this is a example####O O O O
                    sentence = sentence.lower().split()
                    label = label.split()


                # if len(set(label)) == 1 and 'test' not in input_file:
                # 	continue
                assert len(label) == len(sentence), print(sentence, label)
                lines[id] = {'sentence': sentence, 'label': label}
                id += 1
        return lines
    
    def read_txt_o(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            if 'raw_data' in input_file:
            	t = t.split('####')[1]  # 'This=O is=O a=O example=O'
            	sentence = [i.split('=')[0].lower() for i in t.split() if len(i.split('=')[0])]
            	label = [i.split('=')[1] for i in t.split() if len(i.split('=')[0])]
            else:
                sentence, label = t.split('####')  # this is a example####O O O O
                sentence = sentence.lower().split()
                label = label.split()
                if set(label)==set('O'):
                    continue
                if label[0].startswith('#'):
                    sentence.append('#')
                    label[0]=label[0][1:]
                


            # if len(set(label)) == 1 and 'test' not in input_file:
            # 	continue
            assert len(label) == len(sentence), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label}
            id += 1
        return lines


class ABSAProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction and end2end absa ."""
    def _create_examples_gener(self, lines, task_type='ae', set_type=''):
        """Creates examples for the training and dev sets."""
        task_type = task_type.lower()
        assert task_type in {'ae', 'absa','aeoe'}, print('unknow task type ! please choose in [ae, absa]')
        examples = []
        ids = 0
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[i]['sentence']
            label = lines[i]['label']
    
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
            ids += 1
        return examples


    def get_train_examples_source_label(self, data_dir,domain_pair,dataset_split, task_type, fn="/train.txt"):
        """See base class."""
       
        source, _ = domain_pair.split('-')
        if source=='device':

            lines = self.read_txt_o(data_dir + source + '_'+str(dataset_split)+fn)
        else:
            lines = self.read_txt(data_dir + source + '_'+str(dataset_split)+fn)
           
        return self._create_examples_gener(
            lines, task_type= task_type, set_type="train")

    def get_train_examples_da_train(self, data_dir,domain_pair,dataset_split, task_type, fn="/train.txt"):
        """See base class."""
       
        source, target = domain_pair.split('-')
     
        lines = self.read_txt(data_dir + source+'_'+target+'.txt')
           
        return self._create_examples_gener(
            lines, task_type= task_type, set_type="train")

    def get_test_examples_da_test(self, data_dir,domain_pair,dataset_split,task_type, fn="/test.txt"):
        """See base class."""
        _, target = domain_pair.split('-')

        lines = self.read_txt('../splited_data_acl/' + target + '_'+str(dataset_split)+fn)

        return self._create_examples_gener(lines, task_type= task_type, set_type="test")

    def get_test_examples_pseudo_label(self, data_dir,domain_pair,dataset_split,task_type, fn="/train.txt"):
        """See base class."""
        _, target = domain_pair.split('-')

        lines = self.read_txt(data_dir + target + '_'+str(dataset_split)+fn)

        return self._create_examples_gener(lines, task_type= task_type, set_type="test")

    def get_test_examples_generated_filter(self, data_dir,domain_pair,dataset_split,task_type, fn="/da_train.txt"):
        """See base class."""
        source, target = domain_pair.split('-')

        lines = self.read_txt(data_dir + source+'_'+target + '/'+fn)

        return self._create_examples_gener(lines, task_type= task_type, set_type="test")



    def get_labels(self, task_type='ae'):
        """See base class."""
        task_type = task_type.lower()
        assert task_type in {'ae', 'absa','aeoe'}, print('unknow task type ! please choose in [ae, absa]')
        labels = ['O', 'B', 'I'] if task_type == 'ae' else ['O','B-ASP','B-OP','I-ASP','I-OP']
        assert labels[0] == 'O', print('O not in labels! please make sure O in label !')
        return labels

    def ot2bio(self, ts_tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param ts_tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(ts_tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = ts_tag_sequence[i]
            if 'T' not in cur_ts_tag:
                new_ts_sequence.append('O')
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append('I') 
                else:
                    new_ts_sequence.append('B')
            prev_pos = cur_pos
        return new_ts_sequence

    def ot2bio_absa(self, ts_tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param ts_tag_sequence:
        :return: BIO-{POS, NEU, NEG} for end2end absa.
        """
        new_ts_sequence = []
        n_tag = len(ts_tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = ts_tag_sequence[i]
            if 'T' not in cur_ts_tag:
                new_ts_sequence.append('O')
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append('I-%s' % cur_sentiment)  # I 'I-%s' % cur_sentiment
                else:
                    new_ts_sequence.append('B-%s' % cur_sentiment)
            prev_pos = cur_pos
        return new_ts_sequence

    # def _create_examples(self, lines, task_type='ae', set_type=''):
    #     """Creates examples for the training and dev sets."""
    #     task_type = task_type.lower()
    #     assert task_type in {'ae', 'absa','aeoe'}, print('unknow task type ! please choose in [ae, absa]')
    #     examples = []
    #     ids = 0
    #     for i in range(len(lines)):
    #         guid = "%s-%s" % (set_type, ids)
    #         text_a = lines[i]['sentence']
    #         label = self.ot2bio(lines[i]['label']) if task_type == 'ae' else self.ot2bio_absa(lines[i]['label'])
    #         examples.append(
    #             InputExample(guid=guid, text_a=text_a, label=label))
    #         ids += 1
    #     return examples

    def _create_pair_examples(self, lines, task_type='ae', set_type=''):
        """Creates examples for the training and dev sets."""
        task_type = task_type.lower()
        assert task_type in {'ae', 'absa'}, print('unknow task type ! please choose in [ae, absa]')
        examples = []
        ids = 0
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[i]['sentence_a']
            text_b = lines[i]['sentence_b']
            label = self.ot2bio(lines[i]['label']) if task_type == 'ae' else self.ot2bio_absa(lines[i]['label'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            ids += 1
        return examples


def convert_pair_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""  
    PAD_TOKEN_LABEL = -1 
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    label_map['subwords'] = PAD_TOKEN_LABEL

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, labels_a, example.idx_map = tokenizer.subword_tokenize(
            [token.lower() for token in example.text_a], example.label)
        tokens_b, _, _ = tokenizer.subword_tokenize(
            [token.lower() for token in example.text_b], example.label)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            tokens_b = tokens_b[0:(max_seq_length - 2)]
            labels_a = labels_a[0:(max_seq_length - 2)]
        assert len(tokens_a) == len(labels_a) == len(tokens_b), print(tokens_a, tokens_b)


        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]

        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids_a)

        # Zero-pad up to the sequence length.
        while len(input_ids_a) < max_seq_length:
            input_ids_a.append(0)
            input_ids_b.append(0)
            input_mask.append(0)

        assert len(input_ids_a) == max_seq_length
        assert len(input_ids_b) == max_seq_length
        assert len(input_mask) == max_seq_length


        label_id = [PAD_TOKEN_LABEL] * len(input_ids_a)  # -1 is the index to ignore use 0
        # truncate the label length if it exceeds the limit.
        lb = [label_map[label] for label in labels_a]
        if len(lb) > max_seq_length - 2:
            lb = lb[0:(max_seq_length - 2)]
        label_id[1:len(lb) + 1] = lb  # 前后都是-1

        features.append(
            PairInputFeatures(
                input_ids_a=input_ids_a,
                input_ids_b=input_ids_b,
                input_mask=input_mask,
                segment_ids=[0 for _ in range(max_seq_length)],
                label_id=label_id))
    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""  
    PAD_TOKEN_LABEL = -1 
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    label_map['subwords'] = PAD_TOKEN_LABEL

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, labels_a, example.idx_map = tokenizer.subword_tokenize(
            [token.lower() for token in example.text_a], example.label)
        

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            labels_a = labels_a[0:(max_seq_length - 2)]

        assert len(tokens_a) == len(labels_a)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")

        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        output_mask = [1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]
        
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            output_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(output_mask) == max_seq_length


        # label_id = [PAD_TOKEN_LABEL] * len(input_ids)  # -1 is the index to ignore use 0
        # # truncate the label length if it exceeds the limit.
        # lb = [label_map[label] for label in labels_a]
        # if len(lb) > max_seq_length - 2:
        #     lb = lb[0:(max_seq_length - 2)]
        # label_id[1:len(lb) + 1] = lb  # 前后都是-1

        label_id = [label_map[l] for l in labels_a]
        label_padding = [-1] * (max_seq_length-len(label_id))
        label_id += label_padding
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                output_mask=output_mask))
    return features

