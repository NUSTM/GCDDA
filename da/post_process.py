import csv
import nltk
import re
import argparse
import copy

def is_clean_tag(tag_list):
    prev_pos, prev_lab = None, None
    found = True
    for tag in tag_list:
        # if tag != "O":
        #     found = True
        pos, lab = tag[:2], tag[2:]
        if pos not in {"B-", "I-", "E-", "S-", "O"}:
            return False

        if prev_lab is not None:
            if pos in {"I-", "E-"} and lab != prev_lab:  # type conflict
                return False
        prev_pos, prev_lab = pos, lab
    # if prev_pos not in {"E-", "S-", "O"}:  # not end well
    #     return False
    if not found:
        return False
    return True


def is_clean_tok(tok_list):
    
    found = False
    for tok in tok_list:
        if tok != "<unk>":
            found = True
        if tok[:2] in {"B-", "I-", "E-", "S-"}:
            
            return False
    if not found:
        return False
    return True

    
def build_args(parser):
    """Build arguments."""
    parser.add_argument("--source_domain", type=str, default="device")
    parser.add_argument("--target_domain", type=str, default="rest")
    parser.add_argument("--dataset_split", type=str, default="1")
    return parser.parse_args()

args = build_args(argparse.ArgumentParser())
source_domain = args.source_domain
target_domain = args.target_domain
dataset_split = args.dataset_split
print(source_domain, "====================================", target_domain)    
generate_list=[]


with open('./output/'+source_domain+'_'+target_domain+dataset_split+'/test_resultstestest.csv', 'r') as f:
    reader = csv.reader(f)
    d=0
    for i in reader:
        if d==0:
            d+=1
            continue
        d+=1
        re_sentence=i[1]
        label=i[2]
        text=re_sentence.strip().split()[1:]
        label=label.strip().split()[1:]
        assert len(text)==len(label),print(text,label)

        pre_token=''
     

        if not is_clean_tok(text):
            continue
        
        if not is_clean_tag(label):
            continue

        assert len(text)==len(label),print(text,label)

        generate_list.append(' '.join(text)+'####'+' '.join(label))


import os
os.makedirs('./da_data'+dataset_split+'/'+source_domain+'_'+target_domain, exist_ok=True)

with open('./da_data'+dataset_split+'/'+source_domain+'_'+target_domain+'/da_train.txt', 'w') as f:
    f.write('\n'.join(generate_list))