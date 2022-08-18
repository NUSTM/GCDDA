
"""
The core codes of this file i.e., the classes ModelArguments, Seq2SeqTrainer (Trainer API), Seq2SeqTrainingArguments are from https://github.com/huggingface/transformers
"""
import copy
import logging
import os
cur_file = os.path.realpath(__file__)
import math
import re
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import data_util
import transformers
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

from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric,Dataset
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from rouge_score import rouge_scorer

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

# def find_list(indices, data):
#     out = []
#     for i in indices:
#         out = out + [data[i]]
#     return out
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
        # indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        # for data in inputs:
        #     temp_data = find_list(batch_indices, data)
        #     batch_data.append(temp_data)
        yield batch_indices



def evaluate_summary(reference,summary):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference,summary)
    # print(scores)
    return scores["rouge1"]

def get_sub_vocab(tokenizer, vocab):
    new_sub_vocab = []
    for word in vocab:
        sub_word, _ = subwords_tokenizer(tokenizer, word.split(), word.split())
        # if ' '.join(sub_word) in stopwords.words('english') or len(' '.join(sub_word)) == 1:
        #     continue
        # print(sub_word)
       
        new_sub_vocab.append(' '.join(sub_word))
    return new_sub_vocab

def subwords_tokenizer(tokenizer, words, labels):
	# 为了便于对齐，只使用sub word tokenizer并同步扩展标签。
    new_words, new_labels = [], []
    
    for w, l in zip(words, labels):
        w=' '+w
        word_list = tokenizer(w)['input_ids'][1:-1]
        for i,w_ in enumerate(word_list):
            new_words.append(str(w_))
            if i!=0:
                new_labels.append('I-'+l[2:])
            else:
                new_labels.append(l)
    assert len(new_words) == len(new_labels)
    return new_words, new_labels

def subwords_pre_tokenizer(tokenizer, words, labels,words_mask):
	# 为了便于对齐，只使用sub word tokenizer并同步扩展标签。
    new_words, new_labels,new_words_mask = [], [], []
    words=words.split()
    labels=labels.split()
    words_mask=words_mask.split()
    assert len(words) == len(labels)==len(words_mask)
    for i in range(len(words)):
        # if words[i]=='<pad>':
        #     words[i]=' [mask]'
        #     continue
        # if words[i] in {}
        words[i]=' '+words[i]
        # a quad-core 2 .5 ghz cpu
    for j,(w, l) in enumerate(zip(words, labels)):
        flag=False
        word_list = tokenizer(w)['input_ids'][1:-1]
        if words_mask[j]=='<pad>':
            flag=True
        for i,w_ in enumerate(word_list):
            new_words.append(w_)
            if flag:
                
                new_words_mask.append(50267)
            else:
                new_words_mask.append(w_)
            if i!=0:
                if l=='O':
                    new_labels.append(l)
                else:
                    new_labels.append('I-'+l[2:])
            else:
                new_labels.append(l)
    # print(new_words_mask)  
    assert len(new_words) == len(new_labels)==len(new_words_mask)
    return new_words, new_labels,new_words_mask
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus)"
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    source_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    target_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    source_raw_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    source_file_random: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    target_raw_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    opinions_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        # default=1024,
        default=100,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        # default=128,
        default=100,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )




def get_dataset(data_files,tokenizer):
    source_file=data_files["source"]
    target_file=data_files['target']
    source_raw_file=data_files['source_raw']
    target_raw_file=data_files['target_raw']
    source_file_random=data_files["source_random"]

    source_raw_texts=[]
    source_raw_labels=[]

    source_texts_random=[]
    source_labels_random=[]

    target_raw_texts=[]
    target_raw_labels=[]

    source_texts=[]
    source_labels=[]

    target_texts=[]
    target_labels=[]

    label_list={"B-OP": 0,"I-OP": 1,  "B-ASP": 2,  "I-ASP": 3,'O':4,'<target>':50266,'<source>':50265}
    with open(source_raw_file,'r')as fout:
        with open(source_file,'r')as fout2:
            for line,line2 in zip(fout,fout2):

                text=[]
                label=[]

                text_mask=[]
                label_mask=[]

                tokens,tags=line.strip().split('####')
                tokens2,_=line2.strip().split('####')
                
                tokens,tags,tokens_mask=subwords_pre_tokenizer(tokenizer, tokens, tags,tokens2)
             
                label_ids=[]
                for i in tags:
                    label_ids.append(label_list[i])
                text=[label_list["<source>"]]+tokens+[2]
                label=[label_list['O']]+label_ids+[label_list['O']]

                text_mask=[0]+tokens_mask+[2]
                label_mask=[label_list['O']]+label_ids+[label_list['O']]

                assert len(text)==len(label)
                source_raw_texts.append(text)
                source_raw_labels.append(label)

                assert len(text_mask)==len(label_mask)
                source_texts.append(text_mask) 
                source_labels.append(label_mask)
    
    with open(source_raw_file,'r')as fout:   
         raw_file_data=fout.read().splitlines()

    source_raw_texts3=[]
    times=3
    for i in range(times):
        source_raw_texts3.extend(source_raw_texts)
    source_raw_labels3=[]
    for i in range(times):
        source_raw_labels3.extend(source_raw_labels)  

    source_input_texts3=[]
    for i in range(times):
        source_input_texts3.extend(raw_file_data)
    with open(source_file_random,'r')as fout2:
        mask_data=fout2.read().splitlines()
        for line,line2 in zip(source_input_texts3,mask_data):
          

            text=[]
            label=[]

            text_mask=[]
            label_mask=[]

            tokens,tags=line.strip().split('####')
            tokens2,_=line2.strip().split('####')
            
            # is_pretokenized=True
            tokens,tags,tokens_mask=subwords_pre_tokenizer(tokenizer, tokens, tags,tokens2)
    
            label_ids=[]
            for i in tags:
                label_ids.append(label_list[i])
            text=[label_list["<source>"]]+tokens+[2]
            label=[label_list['O']]+label_ids+[label_list['O']]

            text_mask=[0]+tokens_mask+[2]
            label_mask=[label_list['O']]+label_ids+[label_list['O']]



            assert len(text_mask)==len(label_mask)
            source_texts_random.append(text_mask) 
            source_labels_random.append(label_mask)
    

    with open(target_raw_file,'r')as fout:
        with open(target_file,'r')as fout2:
            for line,line2 in zip(fout,fout2):

                text=[]
                label=[]

                text_mask=[]
                label_mask=[]

                tokens,tags=line.strip().split('####')
                tokens_mask,_=line2.strip().split('####')
                
                # is_pretokenized=True
                tokens,tags,tokens_mask=subwords_pre_tokenizer(tokenizer, tokens, tags,tokens_mask)
                
                label_ids=[]
                for i in tags:
                    label_ids.append(label_list[i])
                text=[label_list["<target>"]]+tokens+[2]
                label=[label_list['O']]+label_ids+[label_list['O']]


                text_mask=[0]+tokens_mask+[2]
                label_mask=[label_list['O']]+label_ids+[label_list['O']]

                assert len(text)==len(label)
                target_raw_texts.append(text) 
                target_raw_labels.append(label)

                assert len(text_mask)==len(label_mask)
                target_texts.append(text_mask) 
                target_labels.append(label_mask)

            

    # print(tokenizer.convert_ids_to_tokens(target_texts[2]),tokenizer.convert_ids_to_tokens(target_raw_texts[2]))
    train_data={'mask_data':[],'label_data':[],'domain_label':[],'en_labels':[],'de_labels':[]}
    for i in range(len(source_texts)):
        train_data['mask_data'].append(source_texts[i])
        train_data['label_data'].append(source_raw_texts[i])
        train_data['domain_label'].append(1)
        train_data['en_labels'].append(source_labels[i])
        train_data['de_labels'].append(source_raw_labels[i])
    

    train2_data={'mask_data':[],'label_data':[],'domain_label':[],'en_labels':[],'de_labels':[]}
    for i in range(len(target_texts)):
        train2_data['mask_data'].append(target_texts[i])
        train2_data['label_data'].append(target_raw_texts[i])
        train2_data['domain_label'].append(0)
        train2_data['en_labels'].append(target_labels[i])
        train2_data['de_labels'].append(target_raw_labels[i])

    test_data={'mask_data':[],'label_data':[],'domain_label':[],'en_labels':[],'de_labels':[]}
    for i in range(len(source_texts_random)):
        test_data['mask_data'].append(source_texts_random[i])
        test_data['label_data'].append(source_raw_texts3[i])
        test_data['domain_label'].append(1)
        test_data['en_labels'].append(source_labels_random[i])
        test_data['de_labels'].append(source_raw_labels3[i])
        # test_data.append({'mask_data': source_texts[i], 'label_data': source_raw_texts[i],'domain_label':1})
    
    dataset={}
    dataset["train"]=train_data
    dataset["train2"]=train2_data
    dataset["test"]=test_data

    return dataset
        
        

@dataclass
class DataValidationArguments:
    """
    Arguments pertaining to what parameters we are going to input to our model for validation.
    """
    min_summ_length: Optional[int] = field(
        default=100,
        metadata={
            "help": "The minimum length of the sequence to be generated."
        },
    )
    max_summ_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum length of the sequence to be generated."
        },
    )

    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search."
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=2,
        metadata={
            "help": " If set to int > 0, all ngrams of that size can only occur once."
        },
    )



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


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,DataValidationArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, test_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
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

    tokenizer.add_tokens(["<source>","<target>"," [mask]"],special_tokens = True)

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
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
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
            
        datasets = get_dataset(data_files,tokenizer)

    
   
    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    model.config.decoder_start_token_id=0
    # Get the default prefix if None is passed.


    if training_args.do_train:
        train_dataset = datasets["train"]
        train2_dataset = datasets["train2"]
    
        import math
        train_batch_size=8
        num_train_epochs=3
        num_train_steps = int(math.ceil(len(train_dataset["mask_data"]) / train_batch_size)) * num_train_epochs

        train_features = data_util.convert_examples_to_features(
        train_dataset, 120, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset["mask_data"]))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        all_en_labels = torch.tensor([f.en_labels for f in train_features], dtype=torch.long)
        all_de_labels = torch.tensor([f.de_labels for f in train_features], dtype=torch.long)

        # all_tag_ids = torch.tensor([f.tag_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_en_labels,all_de_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)



        num_train_steps = int(math.ceil(len(train2_dataset["mask_data"]) / train_batch_size)) * num_train_epochs

        train_features = data_util.convert_examples_to_features(
        train2_dataset, 120, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train2_dataset["mask_data"]))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        all_en_labels = torch.tensor([f.en_labels for f in train_features], dtype=torch.long)
        all_de_labels = torch.tensor([f.de_labels for f in train_features], dtype=torch.long)

        # all_tag_ids = torch.tensor([f.tag_id for f in train_features], dtype=torch.long)

        train2_data = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_en_labels,all_de_labels)
        train_sampler = RandomSampler(train2_data)
        train2_dataloader = DataLoader(train2_data, sampler=train_sampler, batch_size=train_batch_size)
    
            

    def get_next_batch(batch):
        return batch.__next__()
    # training_args.do_train=False
    if training_args.do_train:
        def prepare_input_for_training(input_ids, attention_mask, labels, en_labels,de_labels):
            input={}
            input["input_ids"]=input_ids
            input["attention_mask"]=attention_mask
            input["labels"]=labels
            input["en_labels"]=en_labels
            input["de_labels"]=de_labels
            # print(input_ids)
            return input
        import torch.optim as optim

        optimizer = optim.Adam(model.parameters(),lr=5e-5)     
        model.cuda()
        model.train()

        train_steps = len(train_dataloader)
        for e_ in range(num_train_epochs):
            train_loss=0.
            train_loss2=0.
            train_iter = iter(train_dataloader)
            for step in range(train_steps):
                batch = train_iter.next()
                batch = tuple(t.cuda() for t in batch)
                input_ids, attention_mask, labels, en_labels,de_labels = batch
                # input=prepare_input_for_training(input_ids, attention_mask, labels, en_labels,de_labels)
                
                loss,loss2 = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,en_labels=en_labels,de_labels=de_labels)
                # loss = model()
                train_loss+=loss.item()
                train_loss2+=loss2.item()
                z_loss=loss+loss2
                z_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_loss=train_loss/train_steps
            train_loss2=train_loss2/train_steps
            print(f'Epoch: {e_+1:02}')
            print(
                f'\tTrain Loss: {train_loss:.3f}|Train Loss2: {train_loss2:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            torch.save(model, training_args.output_dir+'pytorch_model.bin')

        train_steps = len(train2_dataloader)
        for e_ in range(num_train_epochs):
            train_loss=0.
            train_loss2=0.
            train_iter = iter(train2_dataloader)
            for step in range(train_steps):
                batch = train_iter.next()
                batch = tuple(t.cuda() for t in batch)
                input_ids, attention_mask, labels, en_labels,de_labels = batch
                # input=prepare_input_for_training(input_ids, attention_mask, labels, en_labels,de_labels)
                
                loss,loss2 = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,en_labels=en_labels,de_labels=de_labels)
                # loss = model()
                train_loss+=loss.item()
                train_loss2+=loss2.item()
                z_loss=loss+loss2
                z_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_loss=train_loss/train_steps
            train_loss2=train_loss2/train_steps
            print(f'Epoch: {e_+1:02}')
            print(
                f'\tTrain Loss: {train_loss:.3f}|Train Loss2: {train_loss2:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            torch.save(model, training_args.output_dir+'pytorch_model.bin')

  
    if training_args.do_eval:
        model = torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
        print("\n")
        print("Running Evaluation Script")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # model.cuda()
        model.eval()
        output = []
        inco = 0

        min_length = test_args.min_summ_length
        max_length = test_args.max_summ_length

        # df_test = pd.read_csv(data_args.validation_file)
        test_dataset = datasets["test"]["mask_data"]
        test_label_dataset=datasets["test"]["en_labels"]
        
        iters =len(test_dataset)/32
        import math
        iters=math.ceil(iters)
        
        batch = get_data_batch_one(inputs=test_dataset, batch_size=32, shuffle=False)
        batch_label = get_data_batch_one(inputs=test_label_dataset, batch_size=32, shuffle=False)

               
        def decode_tokens_and_labels(inputs,tokens,labels):
            summary=[]
            summary_labels=[]
            inputs_text=[]
            label_dict={0:' B-OP',1:" I-OP",  2:" B-ASP",  3:" I-ASP",4:' O',5:''}
            for index in range(tokens.size(0)):
                text=""
                label=""
                for i,t in enumerate(tokens[index]):
                    # print(i,tokenizer.decode(t),'?')
                    if tokenizer.decode(t) in {'<pad>'}:
                        break
                    if  (tokenizer.decode(t)==' ' or not tokenizer.decode(t).startswith(' ')) and i==1:
                        # if tokenizer.decode(t) in {'<s>','<target>','</s>',' '}:
                        #     continue
                        text+=tokenizer.decode(t)
                        label+='O'
                        continue
                    if tokenizer.decode(t) in {'<s>','<target>','</s>',' ',' '}:
                        continue
                    if tokenizer.decode(t).startswith(' '):
                        
                        label=label+label_dict[labels[index][i].item()]
                    if  tokenizer.decode(t) in {'.',',','!','?'}:
                        label=label+' O'
                        text=text+' '+tokenizer.decode(t)
                        continue
                    text=text+tokenizer.decode(t)
                assert len(text.strip().split())==len(label.strip().split()),print(text,label)
                summary.append(text)
                summary_labels.append(label)
            for index in range(inputs.size(0)):
                text=""
                for i,t in enumerate(inputs[index]):
                    if tokenizer.decode(t)=='<pad>':
                        break
                    
                    if  tokenizer.decode(t) in {'.',',','!','?'}:
                        text=text+' '+tokenizer.decode(t)
                        continue
                    text=text+tokenizer.decode(t)
                inputs_text.append(text)

            # print(summary,summary_labels)

            return inputs_text,summary,summary_labels

                    

        from tqdm import tqdm
     
        for row in tqdm(range(iters), total=len(range(iters))):
            test_data = get_next_batch(batch)
            test_label_data = get_next_batch(batch_label)
            text=   test_data
            ref = test_data
            
            

            max_length=max([len(t) for t in test_data])+10
            input_ids=torch.ones(len(test_data),max_length, dtype=torch.long).to(device)
            en_labels=torch.empty(len(test_data),max_length, dtype=torch.long).fill_(5).to(device)
            for i in range(len(text)):
                for j in range(len(text[i])):
                    input_ids[i][j]=text[i][j]
                    en_labels[i][j]=test_label_data[i][j]
                    
            with torch.no_grad():
                summary_ids,labels_ids = model.generate(input_ids=input_ids,
                                                    num_beams=1,
                                                    max_length=max_length,
                                                    early_stopping=True,
                                                    en_labels=en_labels
                                                    )
                input_ids,summ,summ_labels=decode_tokens_and_labels(input_ids,summary_ids,labels_ids)
                output.extend(zip(input_ids,summ,summ_labels))
        print("Evaluation Completed")

        actual = [x[0] for x in output]
        generated = [re.sub(r'nnn*n', '',x[1]) for x in output if x[1]!='<pad>']
        generated_labels = [re.sub(r'nnn*n', '',x[2]) for x in output if x[1]!='<pad>']

        df = pd.DataFrame({'Generated Summary':generated,'Generated labels':generated_labels,'Actual Summary':actual})

        csv_output = os.path.join(training_args.output_dir, "test_resultstestest.csv")
        df.to_csv(csv_output)

        print("Evaluation results saved in {}".format(csv_output))




def _mp_fn(index):
    # For xla_spawn (TPUs)
    
    main()


if __name__ == "__main__":
    import os
    
    main()
