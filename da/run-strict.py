
"""
The core codes of this file i.e., the classes ModelArguments, Seq2SeqTrainer (Trainer API), Seq2SeqTrainingArguments are from https://github.com/huggingface/transformers
"""
from datasets import load_dataset, load_metric, Dataset
import torch.optim as optim
from rouge_score import rouge_scorer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.trainer_utils import is_main_process
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    default_data_collator,
    set_seed,
)
import transformers
import data_util
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import sys
import re
import math
import copy
import logging
import os
import math
from model_config import ModelArguments, DataTrainingArguments, DataValidationArguments

cur_file = os.path.realpath(__file__)
logger = logging.getLogger(__name__)


def get_list_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生batch数据
    :param inputs: list数据
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    if shuffle:
        random.shuffle(inputs)
    while True:
        batch_inouts = inputs[0:batch_size]
        inputs = inputs[batch_size:] + inputs[:batch_size]  # 循环移位，以便产生下一个batch
        yield batch_inouts


def get_data_batch_one(inputs, batch_size=None, shuffle=False):
    '''
    产生批量数据batch,非循环迭代
    迭代次数由:iter_nums= math.ceil(sample_nums / batch_size)
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs[0])
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(inputs)
    while True:
        batch_data = []
        cur_nums = len(inputs)
        batch_size = np.where(cur_nums > batch_size, batch_size, cur_nums)
        batch_indices = inputs[0:batch_size]  # 产生一个batch的index
        inputs = inputs[batch_size:]

        yield batch_indices


def evaluate_summary(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores["rouge1"]


def get_sub_vocab(tokenizer, vocab):
    new_sub_vocab = []
    for word in vocab:
        sub_word, _ = subwords_tokenizer(tokenizer, word.split(), word.split())

        new_sub_vocab.append(' '.join(sub_word))
    return new_sub_vocab


def subwords_tokenizer(tokenizer, words, labels):
    # 为了便于对齐，只使用sub word tokenizer并同步扩展标签。
    new_words, new_labels = [], []

    for w, l in zip(words, labels):
        w = ' '+w
        word_list = tokenizer(w)['input_ids'][1:-1]
        for i, w_ in enumerate(word_list):
            new_words.append(str(w_))
            if i != 0:
                new_labels.append('I-'+l[2:])
            else:
                new_labels.append(l)
    assert len(new_words) == len(new_labels)
    return new_words, new_labels


def subwords_pre_tokenizer(tokenizer, words, labels, words_mask):
    # 为了便于对齐，只使用sub word tokenizer并同步扩展标签。

    new_words, new_labels, new_words_mask = [], [], []

    words = words.split()
    labels = labels.split()
    words_mask = words_mask.split()

    assert len(words) == len(labels) == len(words_mask)

    for i in range(len(words)):
        words[i] = ' '+words[i]

    for j, (w, l) in enumerate(zip(words, labels)):
        flag = False
        word_list = tokenizer(w)['input_ids'][1:-1]
        if words_mask[j] == '<pad>':
            flag = True
        for i, w_ in enumerate(word_list):
            new_words.append(w_)
            if flag:
                new_words_mask.append(50267)
            else:
                new_words_mask.append(w_)
            if i != 0:
                if l == 'O':
                    new_labels.append(l)
                else:
                    new_labels.append('I-'+l[2:])
            else:
                new_labels.append(l)
    # print(new_words_mask)
    assert len(new_words) == len(new_labels) == len(new_words_mask)
    return new_words, new_labels, new_words_mask


def get_dataset(data_files, tokenizer):
    '''传入数据
    param1: data_files: 

    including 
    source labeled data_file --data_files['source_raw']
    target weak tagger data_file --data_files['target_raw']
    source masked labeled data_file --data_files["source"]
    target masked weak tagger data_file --data_files['target']
    source masked labeled data_file for inferencing --data_files["source_random"]
    param2: tokenizer  
    '''
    source_masked_file = data_files["source"]
    target_masked_file = data_files['target']
    source_raw_file = data_files['source_raw']
    target_raw_file = data_files['target_raw']
    source_masked_inferencing_file = data_files["source_random"]

    source_mask_texts_docs = []
    source_mask_labels_docs = []
    
    source_raw_texts_docs = []
    source_raw_labels_docs = []

    source_mask_texts_inferencing_docs = []
    source_mask_labels_inferencing_docs = []

    source_raw_texts_inferencing_docs = []
    source_raw_labels_inferencing_docs = []
       
    target_mask_texts_docs = []
    target_mask_labels_docs = []
    
    target_raw_texts_docs = []
    target_raw_labels_docs = []



    label_list = {"B-OP": 0, "I-OP": 1,  "B-ASP": 2,  "I-ASP": 3,
                  'O': 4, '<target>': 50266, '<source>': 50265}
    
    # prepare source training data
    with open(source_raw_file, 'r')as fout:
        with open(source_masked_file, 'r')as fout2:
            for line, line2 in zip(fout, fout2):
                
                tokens, tags = line.strip().split('####')
                tokens_mask, _ = line2.strip().split('####')

                sub_tokens_ids, sub_tags, sub_tokens_mask_ids = subwords_pre_tokenizer(
                    tokenizer, tokens, tags, tokens_mask)

                sub_label_ids = []
                for i in sub_tags:
                    sub_label_ids.append(label_list[i])
                sub_label_ids = [label_list['O']]+sub_label_ids+[label_list['O']]
                
                sub_tokens_ids = [label_list["<source>"]] + sub_tokens_ids + [2]
                sub_tokens_mask_ids = [0]+sub_tokens_mask_ids+[2]

                assert len(sub_tokens_ids) == len(sub_label_ids)
                source_raw_texts_docs.append(sub_tokens_ids)
                source_raw_labels_docs.append(sub_label_ids)

                assert len(sub_tokens_mask_ids) == len(sub_label_ids)
                source_mask_texts_docs.append(sub_tokens_mask_ids)
                source_mask_labels_docs.append(copy.deepcopy(sub_label_ids))

    # prepare target training data
    with open(target_raw_file, 'r')as fout:
        with open(target_masked_file, 'r')as fout2:
            for line, line2 in zip(fout, fout2):

                tokens, tags = line.strip().split('####')
                tokens_mask, _ = line2.strip().split('####')

                sub_tokens_ids, sub_tags, sub_tokens_mask_ids = subwords_pre_tokenizer(
                    tokenizer, tokens, tags, tokens_mask)

                sub_label_ids = []
                for i in sub_tags:
                    sub_label_ids.append(label_list[i])
                sub_label_ids = [label_list['O']]+sub_label_ids+[label_list['O']]
                
                sub_tokens_ids = [label_list["<target>"]] + sub_tokens_ids + [2]
                sub_tokens_mask_ids = [0]+sub_tokens_mask_ids+[2]

                assert len(sub_tokens_ids) == len(sub_label_ids)
                target_raw_texts_docs.append(sub_tokens_ids)
                target_raw_labels_docs.append(sub_label_ids)

                assert len(sub_tokens_mask_ids) == len(sub_label_ids)
                target_mask_texts_docs.append(sub_tokens_mask_ids)
                target_mask_labels_docs.append(copy.deepcopy(sub_label_ids))
    
    # prepare source inferencing data            
    with open(source_raw_file, 'r')as fout:
        raw_file_data = fout.read().splitlines()
        
    times = 3
    source_raw_texts_3_times_docs = []
    for i in range(times):
        source_raw_texts_3_times_docs.extend(raw_file_data)
        
        
    with open(source_masked_inferencing_file, 'r')as fout2:
        source_mask_texts = fout2.read().splitlines()
        for line, line2 in zip(source_raw_texts_3_times_docs, source_mask_texts):

            tokens, tags = line.strip().split('####')
            tokens_mask, _ = line2.strip().split('####')

            sub_tokens_ids, sub_tags, sub_tokens_mask_ids = subwords_pre_tokenizer(
                tokenizer, tokens, tags, tokens_mask)

            sub_label_ids = []
            for i in sub_tags:
                sub_label_ids.append(label_list[i])
            sub_label_ids = [label_list['O']]+sub_label_ids+[label_list['O']]
            
            sub_tokens_ids = [label_list["<source>"]] + sub_tokens_ids + [2]
            sub_tokens_mask_ids = [0]+sub_tokens_mask_ids+[2]

            assert len(sub_tokens_ids) == len(sub_label_ids)
            source_raw_texts_inferencing_docs.append(sub_tokens_ids)
            source_raw_labels_inferencing_docs.append(sub_label_ids)

            assert len(sub_tokens_mask_ids) == len(sub_label_ids)
            source_mask_texts_inferencing_docs.append(sub_tokens_mask_ids)
            source_mask_labels_inferencing_docs.append(copy.deepcopy(sub_label_ids))


    source_train_data = {'mask_data': [], 'label_data': [], 'en_labels': [], 'de_labels': []}
    for i in range(len(source_mask_texts_docs)):
        source_train_data['mask_data'].append(source_mask_texts_docs[i])
        source_train_data['label_data'].append(source_raw_texts_docs[i])
        source_train_data['en_labels'].append(source_mask_labels_docs[i])
        source_train_data['de_labels'].append(source_raw_labels_docs[i])

    target_train_data = {'mask_data': [], 'label_data': [], 'en_labels': [], 'de_labels': []}
    for i in range(len(target_mask_texts_docs)):
        target_train_data['mask_data'].append(target_mask_texts_docs[i])
        target_train_data['label_data'].append(target_raw_texts_docs[i])
        target_train_data['en_labels'].append(target_mask_labels_docs[i])
        target_train_data['de_labels'].append(target_raw_labels_docs[i])

    test_data = {'mask_data': [], 'label_data': [], 'en_labels': [], 'de_labels': []}
    for i in range(len(source_mask_texts_inferencing_docs)):
        test_data['mask_data'].append(source_mask_texts_inferencing_docs[i])
        test_data['label_data'].append(source_raw_texts_inferencing_docs[i])
        test_data['en_labels'].append(source_mask_labels_inferencing_docs[i])
        test_data['de_labels'].append(source_raw_labels_inferencing_docs[i])
        # test_data.append({'mask_data': source_texts[i], 'label_data': source_raw_texts[i],'domain_label':1})

    dataset = {}
    dataset["train_source"] = source_train_data
    dataset["train_target"] = target_train_data
    dataset["test"] = test_data

    return dataset

def convert_data_ids_to_model_inputs(train_dataset,tokenizer):
    
    train_batch_size = 8
    num_train_epochs = 3
    max_length = 200
    
    num_train_steps = int(math.ceil(
        len(train_dataset["mask_data"]) / train_batch_size)) * num_train_epochs

    train_features = data_util.convert_examples_to_features(
        train_dataset, max_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset["mask_data"]))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in train_features], dtype=torch.long)
    all_labels = torch.tensor(
        [f.labels for f in train_features], dtype=torch.long)
    all_en_labels = torch.tensor(
        [f.en_labels for f in train_features], dtype=torch.long)
    all_de_labels = torch.tensor(
        [f.de_labels for f in train_features], dtype=torch.long)

    # all_tag_ids = torch.tensor([f.tag_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(
        all_input_ids, all_attention_mask, all_labels, all_en_labels, all_de_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)  
    return train_dataloader,num_train_steps

def train_model(train_dataloader,num_train_steps,model,training_args):
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    model.cuda()
    model.train()

    train_steps = len(train_dataloader)
    for e_ in range(num_train_steps):
        train_loss = 0.
        train_loss2 = 0.
        train_iter = iter(train_dataloader)
        for step in range(train_steps):
            batch = train_iter.next()
            batch = tuple(t.cuda() for t in batch)
            input_ids, attention_mask, labels, en_labels, de_labels = batch

            loss, loss2 = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels, en_labels=en_labels, de_labels=de_labels)
            train_loss += loss.item()
            train_loss2 += loss2.item()
            z_loss = loss+loss2
            z_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = train_loss/train_steps
        train_loss2 = train_loss2/train_steps
        print(f'Epoch: {e_+1:02}')
        print(
            f'\tTrain Loss: {train_loss:.3f}|Train Loss2: {train_loss2:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        torch.save(model, training_args.output_dir+'pytorch_model.bin')

def decode_tokens_and_labels(inputs, tokens, labels,tokenizer):
    summary = []
    summary_labels = []
    inputs_text = []
    label_dict = {0: ' B-OP', 1: " I-OP",
                    2: " B-ASP",  3: " I-ASP", 4: ' O', 5: ''}
    for index in range(tokens.size(0)):
        text = ""
        label = ""
        for i, t in enumerate(tokens[index]):
            # print(i,tokenizer.decode(t),'?')
            if tokenizer.decode(t) in {'<pad>'}:
                break
            if (tokenizer.decode(t) == ' ' or not tokenizer.decode(t).startswith(' ')) and i == 1:
                # if tokenizer.decode(t) in {'<s>','<target>','</s>',' '}:
                #     continue
                text += tokenizer.decode(t)
                label += 'O'
                continue
            if tokenizer.decode(t) in {'<s>', '<target>', '</s>', ' ', ' '}:
                continue
            if tokenizer.decode(t).startswith(' '):

                label = label+label_dict[labels[index][i].item()]
            if tokenizer.decode(t) in {'.', ',', '!', '?'}:
                label = label+' O'
                text = text+' '+tokenizer.decode(t)
                continue
            text = text+tokenizer.decode(t)
            
        assert len(text.strip().split()) == len(
            label.strip().split()), print(text, label)
        
        summary.append(text)
        summary_labels.append(label)
    for index in range(inputs.size(0)):
        text = ""
        for i, t in enumerate(inputs[index]):
            if tokenizer.decode(t) == '<pad>':
                break

            if tokenizer.decode(t) in {'.', ',', '!', '?'}:
                text = text+' '+tokenizer.decode(t)
                continue
            text = text+tokenizer.decode(t)
        inputs_text.append(text)

    # print(summary,summary_labels)

    return inputs_text, summary, summary_labels 

def get_next_batch(batch):
    return batch.__next__()  
         
def evaluation(model,test_dataset,test_label_dataset,device,test_args):
    
    output = []
    iters = len(test_dataset)/32
    iters = math.ceil(iters)
    max_length = test_args.max_summ_length
    
    batch = get_data_batch_one(
        inputs=test_dataset, batch_size=32, shuffle=False)
    batch_label = get_data_batch_one(
        inputs=test_label_dataset, batch_size=32, shuffle=False)
    
    for row in tqdm(range(iters), total=len(range(iters))):
        test_data = get_next_batch(batch)
        test_label_data = get_next_batch(batch_label)
        text = test_data

        max_length = max([len(t) for t in test_data])+10
        input_ids = torch.ones(
            len(test_data), max_length, dtype=torch.long).to(device)
        en_labels = torch.empty(
            len(test_data), max_length, dtype=torch.long).fill_(5).to(device)
        for i in range(len(text)):
            for j in range(len(text[i])):
                input_ids[i][j] = text[i][j]
                en_labels[i][j] = test_label_data[i][j]

        with torch.no_grad():
            # We modify the code on /transformers/geration_utils.py greedy_search() and on /transformers/models/bart/modeling_bart.py BartForConditionalGeneration()
            summary_ids, labels_ids = model.generate(input_ids=input_ids,
                                                        num_beams=1,
                                                        max_length=max_length,
                                                        early_stopping=True,
                                                        en_labels=en_labels
                                                        )
            input_ids, summ, summ_labels = decode_tokens_and_labels(
                input_ids, summary_ids, labels_ids)
            output.extend(zip(input_ids, summ, summ_labels))
    return output

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               DataValidationArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, test_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, test_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(
            training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    import random
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    set_seed(training_args.seed)

    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer.add_tokens(
        ["<source>", "<target>", " [mask]"], special_tokens=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name,
                                data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.source_file is not None:
            data_files["source"] = data_args.source_file
        if data_args.target_file is not None:
            data_files["target"] = data_args.target_file
        if data_args.source_raw_file is not None:
            data_files["source_raw"] = data_args.source_raw_file
        if data_args.target_raw_file is not None:
            data_files["target_raw"] = data_args.target_raw_file
        if data_args.source_file_random is not None:
            data_files["source_random"] = data_args.source_file_random

        datasets = get_dataset(data_files, tokenizer)

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")
    model.config.decoder_start_token_id = 0
    # Get the default prefix if None is passed.
            
    if training_args.do_train:
        
        source_train_dataset = datasets["train_source"]
        target_train_dataset = datasets["train_target"]
        
        source_train_dataloader,source_num_train_steps = convert_data_ids_to_model_inputs(source_train_dataset,tokenizer)
        target_train_dataloader,target_num_train_steps = convert_data_ids_to_model_inputs(target_train_dataset,tokenizer)
        
        train_model(source_train_dataloader,source_num_train_steps,model,training_args)
        train_model(target_train_dataloader,target_num_train_steps,model,training_args)
    

    if training_args.do_eval:
        
        model = torch.load(os.path.join(
            training_args.output_dir, "pytorch_model.bin"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
                
        print("\n")
        print("Running Evaluation Script")
        
        test_dataset = datasets["test"]["mask_data"]
        test_label_dataset = datasets["test"]["en_labels"]
     
           
        output = evaluation(model,test_dataset,test_label_dataset,device,test_args)

        actual = [x[0] for x in output]
        generated = [re.sub(r'nnn*n', '', x[1])
                     for x in output if x[1] != '<pad>']
        generated_labels = [re.sub(r'nnn*n', '', x[2])
                            for x in output if x[1] != '<pad>']
        df = pd.DataFrame({'Generated Summary': generated,
                           'Generated labels': generated_labels, 'Actual Summary': actual})
        csv_output = os.path.join(
            training_args.output_dir, "test_resultstestest.csv")
        df.to_csv(csv_output)
        
        print("Evaluation Completed")
        print("Evaluation results saved in {}".format(csv_output))

if __name__ == "__main__":
    import os

    main()
