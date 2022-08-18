import argparse
from collections import defaultdict
import torch
import numpy as np
import random
import os
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)           # 为CPU设置随机种子
torch.cuda.manual_seed(42)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(42)
def build_args(parser):
    """Build arguments."""
    parser.add_argument("--source_domain", type=str, default="device")
    parser.add_argument("--target_domain", type=str, default="rest")
    parser.add_argument("--dataset_split", type=int, default=1)
    return parser.parse_args()

def max_match(text,tags,attributes,aspect_terms):
    '''
    text: list(list(str)) 句子列表
    vocab: 词典
    基于词典的最大匹配: 找到text中所包含的来自于aspects的词
    '''
   
    
        
    line= text
    covered = set()
    covered2 = set()
    for start in range(len(line)):
        if start in covered | covered2:
            continue
        # private mask

        for end in range(len(line) - 1, start - 1, -1):
            if ' '.join(line[start: end + 1]).lower() in attributes:
                # print(' '.join(line[start: end + 1]))
                # if end != start:
                    # muliti_ += 1
                if any([i in covered | covered2 for i in range(start, end + 1)]):
                    continue
                # prop=random.randint(0,1)
                
                
                # if prop>0.8 and set(tags[start:end + 1])==set(['O']):
                #     continue
                prop=random.random()
                
                # set(tags[start:end + 1]).issubset(set(['O','B-OP','I-OP']))
                if prop>0.6 and set(tags[start:end + 1])==set(['O']):
                    # covered2.update([i for i in range(start, end + 1)])
                    continue
                covered.update([i for i in range(start, end + 1)])
                break
        
        # if start in covered2:
        #     continue
        # # private mask

        # for end in range(len(line) - 1, start - 1, -1):
        #     if ' '.join(line[start: end + 1]).lower() in aspect_terms:
        #         # print(' '.join(line[start: end + 1]))
        #         # if end != start:
        #             # muliti_ += 1
        #         if any([i in covered2 for i in range(start, end + 1)]):
        #             continue
        #         covered2.update([i for i in range(start, end + 1)])
        #         break
        # if start not in covered and not line[start].startswith('##'):
        #     end = start
        #     while end + 1 < len(line):
        #         if not line[end + 1].startswith('##'):
        #             break
        #         end += 1
        #     if ' '.join(line[start: end + 1]) in opinions:

        #         line[start: end + 1] = ['<pad>' for i in range(start, end + 1)]
    for i in covered:
        line[i] = '<pad>'

    
    label_index_lists=[]
    label_index=[]
    flag=False
    
    for i in range(len(line)):
        if tags[i]=='O':
            if flag==True:
                flag=False
                label_index_lists.append(label_index)
                label_index=[]
            continue
        elif tags[i].startswith('B-'):
            if flag==True:
                label_index_lists.append(label_index)
                label_index=[]
            else:
                flag=True 
                label_index.append(i)
            
        elif tags[i].startswith('I-'):
            if flag==False:
                flag=True
                label_index_lists.append(label_index)
                label_index=[]
            label_index.append(i)

    for l in label_index_lists:
        if len(set(l) & covered)!=0:
            
            for i in l:
                line[i]='<pad>'

def max_match_no_random(text,tags,attributes,aspect_terms,seed):
    '''
    text: list(list(str)) 句子列表
    vocab: 词典
    基于词典的最大匹配: 找到text中所包含的来自于aspects的词
    '''
   
    
        
    line= text
    covered = set()
    covered2 = set()
    for start in range(len(line)):
        if start in covered | covered2:
            continue
        # private mask

        for end in range(len(line) - 1, start - 1, -1):
            if ' '.join(line[start: end + 1]).lower() in attributes:
                # print(' '.join(line[start: end + 1]))
                # if end != start:
                    # muliti_ += 1
                if any([i in covered | covered2 for i in range(start, end + 1)]):
                    continue
                prop=random.random()
                
                # set(tags[start:end + 1]).issubset(set(['O','B-OP','I-OP']))
                if prop>0.6 and set(tags[start:end + 1])==set(['O']):
                    # covered2.update([i for i in range(start, end + 1)])
                    continue
                covered.update([i for i in range(start, end + 1)])
                break
        
        # if start in covered2:
        #     continue
        # # private mask
    
        # for end in range(len(line) - 1, start - 1, -1):
        #     if ' '.join(line[start: end + 1]).lower() in aspect_terms:
        #         # print(' '.join(line[start: end + 1]))
        #         # if end != start:
        #             # muliti_ += 1
        #         if any([i in covered2 for i in range(start, end + 1)]):
        #             continue
        #         covered2.update([i for i in range(start, end + 1)])
        #         break
        # if start not in covered and not line[start].startswith('##'):
        #     end = start
        #     while end + 1 < len(line):
        #         if not line[end + 1].startswith('##'):
        #             break
        #         end += 1
        #     if ' '.join(line[start: end + 1]) in opinions:

        #         line[start: end + 1] = ['<pad>' for i in range(start, end + 1)]
    for i in covered:
        line[i] = '<pad>'

    
    label_index_lists=[]
    label_index=[]
    flag=False
    
    for i in range(len(line)):
        if tags[i]=='O':
            if flag==True:
                flag=False
                label_index_lists.append(label_index)
                label_index=[]
            continue
        elif tags[i].startswith('B-'):
            if flag==True:
                label_index_lists.append(label_index)
                label_index=[]
            else:
                flag=True 
                label_index.append(i)
            
        elif tags[i].startswith('I-'):
            if flag==False:
                flag=True
                label_index_lists.append(label_index)
                label_index=[]
            label_index.append(i)

    for l in label_index_lists:
        if len(set(l) & covered)!=0:
            
            for i in l:
                line[i]='<pad>'

def ot2bio_absa(ts_tag_sequence):
    """
    ot2bio function for ts tag sequence
    :param ts_tag_sequence:
    :return: BIO-{POS, NEU, NEG} for end2end absa.
    """
    new_ts_sequence = []
    n_tag = len(ts_tag_sequence)
    pre_tag = 'O'
    for i in range(n_tag):
        tag = ts_tag_sequence[i]
        if 'T' not in tag:
            new_ts_sequence.append('O')
            cur_tag = 'O'
        else:
            cur_tag, sentiment = tag.split('-')
            if pre_tag == 'O':
                new_ts_sequence.append('B-%s' % sentiment)
            else:
                new_ts_sequence.append('I-%s' % sentiment)
        pre_tag = cur_tag
    return new_ts_sequence

"""
python make_ngram_attribute_vocab.py [vocab] [corpus1] [corpus2] r

subsets a [vocab] file by finding the words most associated with
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
uses ngrams
"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk import ngrams
from tqdm import tqdm

class NgramSalienceCalculator(object):
    def __init__(self, pre_corpus, post_corpus, tokenize):
        self.vectorizer = CountVectorizer(tokenizer=tokenize)

        pre_count_matrix = self.vectorizer.fit_transform(pre_corpus)
        self.pre_vocab = self.vectorizer.vocabulary_
        self.pre_counts = np.sum(pre_count_matrix, axis=0)
        self.pre_counts = np.squeeze(np.asarray(self.pre_counts))
        self.pre_counts_all = np.sum(self.pre_counts)

        post_count_matrix = self.vectorizer.fit_transform(post_corpus)
        self.post_vocab = self.vectorizer.vocabulary_
        self.post_counts = np.sum(post_count_matrix, axis=0)
        self.post_counts = np.squeeze(np.asarray(self.post_counts))
        self.post_counts_all = np.sum(self.post_counts)

    def salience(self, feature, attribute='pre', lmbda=1):
        assert attribute in ['pre', 'post']
        if feature not in self.pre_vocab:
            pre_count = 0.0
        else:
            pre_count = self.pre_counts[self.pre_vocab[feature]]

        if feature not in self.post_vocab:
            post_count = 0.0
        else:
            post_count = self.post_counts[self.post_vocab[feature]]
        if attribute == 'pre':
            return (pre_count + lmbda) / (post_count + lmbda)
        else:
            return (post_count + lmbda) / (pre_count + lmbda)
        # if attribute == 'pre':
        #     return ((pre_count + lmbda)/self.pre_counts_all) / ((post_count + lmbda)/self.post_counts_all)
        # else:
        #     return ((post_count + lmbda)/self.post_counts_all) / ((pre_count + lmbda)/self.pre_counts_all)




def tokenize(text):
    text = text.split()
    grams = []
    for i in range(1, 5):
        i_grams = [
            " ".join(gram)
            for gram in ngrams(text, i)
        ]
        grams.extend(i_grams)
    return grams


def read(path):
	with open(path, 'r', encoding='utf-8') as fp:
		return set(fp.read().splitlines())


Attributes_dict=defaultdict(int)
def calculate_attribute_markers(corpus):
    for sentence in tqdm(corpus):
        for i in range(1, 5):
            i_grams = ngrams(sentence.split(), i)
            joined = [
                " ".join(gram)
                for gram in i_grams
            ]
            for gram in joined:
                negative_salience = sc.salience(gram, attribute='pre')
                positive_salience = sc.salience(gram, attribute='post')
                if max(negative_salience, positive_salience) > r:
                    # print(gram, negative_salience, positive_salience)
                    # print(gram)
                    if negative_salience>positive_salience:
                        Attributes_dict[gram]=max(negative_salience, positive_salience)
                    else:
                        Attributes_dict[gram]=-max(negative_salience, positive_salience)




    
            
    

args = build_args(argparse.ArgumentParser())
source_domain = args.source_domain
target_domain = args.target_domain
dataset_split = args.dataset_split
print(source_domain, "====================================", target_domain)

count=0

texts_t=[]
texts_t_dp=[]
texts_s=[]
tags_t=[]
tags_t_dp=[]
tags_s=[]


with open('./splited_data_acl/' + source_domain + '_'+str(dataset_split)+'/train.txt','r')as fout:
    for line in fout:
        linear_data=[]
        tokens=[]
        tags=[]
        text,label=line.strip().split("####")
        tokens=text.split()
        tags=label.split()
        if source_domain=='device':
            if set(tags)==set(['O']):
                continue
        assert len(tokens)==len(tags)
        texts_s.append(' '.join(tokens))
        tags_s.append(' '.join(tags))


# with open('./splited_data_dp_iter_all/' + target_domain + '_'+str(split_index+1)+'.txt','r')as fout:
#     for line in fout:
#         linear_data=[]
#         tokens=[]
#         tags=[]
#         text,label=line.strip().split("####")
#         tokens=text.split()
#         tags=label.split()
#         assert len(tokens)==len(tags)
#         texts_t.append(' '.join(tokens))
#         tags_t.append(' '.join(tags))

# with open('./raw_data/%s_train.txt'%target_domain,'r')as fout:
#     for line in fout:
#         linear_data=[]
#         tokens=[]
#         tags=[]
#         _,text=line.strip().split("####")
#         words=text.split()
#         for word in words:
#             if len(word.split("="))>2:
#                 tokens.append('=')
#                 tags.append('O')
#             else:
#                 token,label=word.split("=")
#                 # token = normalize_tok(token)
#                 tokens.append(token.lower())
#                 tags.append(label)
#         tags=ot2bio_absa(tags)
#         assert len(tokens)==len(tags)
#         texts_t.append(' '.join(tokens))
#         tags_t.append(' '.join(tags))

# with open('./co_guess_data52/%s_%s.txt'%(source_domain,target_domain),'r')as fout:
#     for line in fout:
#         tok_list, tag_list = [], []
#         tokens,labels = line.strip().split('####')
#         tokens=tokens.split()
#         labels=labels.split()
#         for i in range(len(tokens)):
#             if tokens[i].startswith("##"):
#                 tok_list[-1]=tok_list[-1]+tokens[i][2:]
#             else:
#                 tok_list.append(tokens[i])
#                 tag_list.append(labels[i])
#         texts_t.append(' '.join(tok_list))
#         tags_t.append(' '.join(tag_list))

def max_match_op(line, label,dict_list,muliti_):
    '''
    text: list(list(str)) 句子列表
    vocab: 词典
    基于词典的最大匹配: 找到text中所包含的来自于aspects的词
    '''
    
    
    covered = set()
    
    for start in range(len(line)):
        if start in covered:
            continue

        # private mask
        for end in range(len(line) - 1, start - 1, -1):
            if ' '.join(line[start: end + 1]).lower() in dict_list:
                # print(' '.join(line[start: end + 1]))
                # if end != start:
                #     muliti_ += 1
                if any([i in covered for i in range(start, end + 1)]):
                    continue
                covered.update([i for i in range(start, end + 1)])
                if label[start]=='O':
                    if start>=1 and label[start-1][2:]=='OP':
                        label[start]='I-OP'
                    else:
                        label[start]='B-OP'
                    muliti_+=1
                    
                break
    return muliti_  
# base_lexicons = positive_lexicon, negative_lexicon, neutral_lexicon = read('./double_prop_tag-aeoe/opinion-lexicon-English/positive_opinions.txt'),  read('./double_prop_tag-aeoe/opinion-lexicon-English/negative_opinions.txt'), set()
# # base_lexicons = positive_lexicon, negative_lexicon, neutral_lexicon = get_base_lexicons('./double_prop/lexicon.txt')
# base_lexicons = set.union(*base_lexicons)
# base_lexicons=dict.fromkeys(base_lexicons, 0)
muliti_=0
with open('./split'+str(dataset_split)+'_sl/%s-%s/pre.txt'%(source_domain,target_domain),'r')as fout:
    for line in fout:
        tok_list, tag_list = [], []
        tokens,labels,_ = line.strip().split('***')
        tokens=tokens.split()
        labels=labels.split()
        for i in range(len(tokens)):
            if tokens[i].startswith("##"):
                tok_list[-1]=tok_list[-1]+tokens[i][2:]
            else:
                tok_list.append(tokens[i])
                tag_list.append(labels[i])
        # muliti_=max_match_op(tok_list,tag_list,base_lexicons,muliti_)
        # if set(tag_list)==set(['O']):
        #     continue
        texts_t.append(' '.join(tok_list))
        tags_t.append(' '.join(tag_list))

# with open('./splited_data_small-100/' + target_domain + '_'+str(dataset_split)+'/train.txt','r')as fout:
#     for line in fout:
#         linear_data=[]
#         tokens=[]
#         tags=[]
#         text,label=line.strip().split("####")
#         tokens=text.split()
#         tags=label.split()
#         # if source_domain=='device':
#         #     if set(tags)==set(['O']):
#         #         continue
#         assert len(tokens)==len(tags)
#         texts_t.append(' '.join(tokens))
#         tags_t.append(' '.join(tags))


# print(muliti_)
def get_features(text_list, labels_list):
    # 从标注样例中抽取属性词
    aspects = set()
    for index in range(len(text_list)):
        line, label = text_list[index].split(), labels_list[index].split()
        covered = set()
        for start in range(len(line)):
            if start in covered:
                continue
                # ground truth
            if label[start] != 'O':
                left, right = start, start
                while left - 1 >= 0:
                    if label[left - 1] == 'O':
                        break
                    left -= 1
                while right + 1 < len(line):
                    if label[right + 1] == 'O':
                        break
                    right += 1
                aspects.add(' '.join(line[left: right + 1]))
                covered.update([i for i in range(left, right + 1)])
    return aspects
# the salience ratio
r = float(10)


sc = NgramSalienceCalculator(texts_s, texts_t, tokenize)

print("marker", "negative_score", "positive_score")


calculate_attribute_markers(texts_s)
calculate_attribute_markers(texts_t)
# 
#     fin.write('\n'.join(list(set(Attributes))))
os.makedirs('./attribute_balance/', exist_ok=True)
with open('./attribute_balance/%s_%s_attributes_dict.txt'%(source_domain,target_domain),'w')as fin:
    for k,v in Attributes_dict.items():
        fin.write('%s'%(k))
        fin.write('\n')




target_features=get_features(texts_t, tags_t)


with open('./attribute_balance/%s_%s_attributes_dict.txt'%(source_domain,target_domain),'r')as fin:
    attributes =set(fin.read().lower().splitlines())

# aspect_terms=private_target_features|private_source_features
aspect_terms=''
# private_target_features=[]
# private_source_features=[]

# target raw_data dp+bert-e
# raw_data_t=[]
raw_data_t_noli=[]
for j in range(len(texts_t)):
    tokens=texts_t[j].split()
    tags=tags_t[j].split()
    raw_data_t_noli.append(' '.join(tokens)+'####'+' '.join(tags))
    # linear_data=[]
    
    # for i in range(len(tokens)):
    #     if tags[i]!='O':

    #         linear_data+=[tags[i],tokens[i]]
    #     else:
    #         linear_data+=[tokens[i]]
    # raw_data_t.append(linear_data)

# for j in range(len(texts_t_dp)):
#     tokens=texts_t_dp[j].split()
#     tags=tags_t_dp[j].split()
    
#     linear_data=[]
    
#     for i in range(len(tokens)):
#         if tags[i]!='O':

#             linear_data+=[tags[i],tokens[i]]
#         else:
#             linear_data+=[tokens[i]]
#     raw_data_t.append(linear_data)

# target mask_data dp+bert-e
# mask_data_t=[]
mask_data_t_noli=[]
for j in range(len(texts_t)):
    tokens=texts_t[j].split()
    tags=tags_t[j].split()
    
    max_match(tokens,tags,attributes,aspect_terms)
    mask_data_t_noli.append(' '.join(tokens)+'####'+' '.join(tags))
    # linear_data=[]
    
    # for i in range(len(tokens)):
    #     if tags[i]!='O':

    #         linear_data+=[tags[i],tokens[i]]
    #     else:
    #         linear_data+=[tokens[i]]
    # mask_data_t.append(linear_data)

# for j in range(len(texts_t_dp)):
#     tokens=texts_t_dp[j].split()
#     tags=tags_t_dp[j].split()
    
#     max_match(tokens,tags,attributes,aspect_terms)
#     linear_data=[]
    
#     for i in range(len(tokens)):
#         if tags[i]!='O':

#             linear_data+=[tags[i],tokens[i]]
#         else:
#             linear_data+=[tokens[i]]
#     mask_data_t.append(linear_data)

# source raw_data 

# raw_data_s=[]
raw_data_s_noli=[]
for j in range(len(texts_s)):
    tokens=texts_s[j].split()
    tags=tags_s[j].split()
    linear_data=[]
    raw_data_s_noli.append(' '.join(tokens)+'####'+' '.join(tags))
    # for i in range(len(tokens)):
    #     if tags[i]!='O':

    #         linear_data+=[tags[i],tokens[i]]
    #     else:
    #         linear_data+=[tokens[i]]
    # raw_data_s.append(linear_data)

# source mask_data

# mask_data_s=[]
mask_data_s_noli=[]
for j in range(len(texts_s)):
    tokens=texts_s[j].split()
    tags=tags_s[j].split()
    
    max_match(tokens,tags,attributes,aspect_terms)
    # linear_data=[]
    mask_data_s_noli.append(' '.join(tokens)+'####'+' '.join(tags))
    # for i in range(len(tokens)):
    #     if tags[i]!='O':

    #         linear_data+=[tags[i],tokens[i]]
    #     else:
    #         linear_data+=[tokens[i]]
    # mask_data_s.append(linear_data)
    
# mask_data_s_random=[]
mask_data_s_noli_random=[]
times=5
for time in range(times):
    for j in range(len(texts_s)):
        tokens=texts_s[j].split()
        tags=tags_s[j].split()
        
        max_match_no_random(tokens,tags,attributes,aspect_terms,time)
        # linear_data=[]
        mask_data_s_noli_random.append(' '.join(tokens)+'####'+' '.join(tags))
        # for i in range(len(tokens)):
        #     if tags[i]!='O':

        #         linear_data+=[tags[i],tokens[i]]
        #     else:
        #         linear_data+=[tokens[i]]
        # mask_data_s_random.append(linear_data)



# raw_data_s_3times=[]
raw_data_s_noli_3times=[]
times=5
for time in range(times):
    
    for j in range(len(texts_s)):
        tokens=texts_s[j].split()
        tags=tags_s[j].split()
        # linear_data=[]
        raw_data_s_noli_3times.append(' '.join(tokens)+'####'+' '.join(tags))
        # for i in range(len(tokens)):
        #     if tags[i]!='O':

        #         linear_data+=[tags[i],tokens[i]]
        #     else:
        #         linear_data+=[tokens[i]]
        # raw_data_s_3times.append(linear_data)


# os.makedirs('./lin_data-raw_data-noli-3times/', exist_ok=True)
# with open('./lin_data-raw_data-noli-3times/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
#     for line in raw_data_s_noli_3times:
#         fin.write(line)
#         fin.write('\n')


os.makedirs('./masked_target_pseudo_label/', exist_ok=True)
with open('./masked_target_pseudo_label/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
    for line in mask_data_t_noli:
        fin.write(line)
        fin.write('\n')

os.makedirs('./target_pseudo_label/', exist_ok=True)
with open('./target_pseudo_label/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
    for line in raw_data_t_noli:
        fin.write(line)
        fin.write('\n')

os.makedirs('./masked_source_label/', exist_ok=True)
with open('./masked_source_label/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
    for line in mask_data_s_noli:
        fin.write(line)
        fin.write('\n')
os.makedirs('./source_label/', exist_ok=True)
with open('./source_label/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
    for line in raw_data_s_noli:
        fin.write(line)
        fin.write('\n') 

os.makedirs('./source_label-random/', exist_ok=True)
with open('./source_label-random/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
    for line in mask_data_s_noli_random:
        fin.write(line)
        fin.write('\n')






# os.makedirs('./mask_data-dp+bert-e-iter_all/', exist_ok=True)
# with open('./mask_data-dp+bert-e-iter_all/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
#     for line in mask_data_t:
#         fin.write(' '.join(line))
#         fin.write('\n')
# os.makedirs('./mask_data-raw_data/', exist_ok=True)
# with open('./mask_data-raw_data/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
#     for line in mask_data_s:
#         fin.write(' '.join(line))
#         fin.write('\n')
# os.makedirs('./lin_data-raw_data/', exist_ok=True)
# with open('./lin_data-raw_data/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
#     for line in raw_data_s:
#         fin.write(' '.join(line))
#         fin.write('\n')

# os.makedirs('./lin_data-dp+bert-e-iter_all/', exist_ok=True)
# with open('./lin_data-dp+bert-e-iter_all/'+source_domain+'_'+target_domain+'_'+str(dataset_split)+'_lin.txt','w')as fin:
#     for line in raw_data_t:
#         fin.write(' '.join(line))
#         fin.write('\n')




