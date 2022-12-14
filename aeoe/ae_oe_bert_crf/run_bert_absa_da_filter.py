# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.append('../')
import logging
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import BertPreTrainedModel, BertModel
from optimization import BertAdam
import math
import data_utils
from data_utils import ABSATokenizer
import modelconfig
from eval import eval_result, eval_ts_result,print_valid_result
from collections import namedtuple
from tqdm import tqdm
from net.bert_ner import Bert_CRF

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     filename='new.log',
#                     filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     #a是追加模式，默认如果不写的话，就是追加模式
#                     level=logging.INFO)
logger = logging.getLogger(__name__)

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def train(args):
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels(args.task_type)

    model = Bert_CRF.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model],
              num_tag = len(label_list))

    model.cuda()

    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    train_examples = processor.get_train_examples_source_label(args.data_dir,args.domain_pair,args.dataset_split, args.task_type)
    num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in train_features], dtype=torch.long)
    # all_tag_ids = torch.tensor([f.tag_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_output_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.do_valid:
        valid_examples = processor.get_dev_examples(args.data_dir, args.task_type)
        valid_features = data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer)

        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_output_mask = torch.tensor([f.output_mask for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_label_ids,valid_all_output_mask)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)

        max_ae=0.
        valid_losses = []

    model.cuda()
    
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True and not k.startswith('crf')]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps  # num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total
                         )
    crf_grouped_parameters=model.crf.parameters()
    crf_param_optimizer = BertAdam(crf_grouped_parameters,
                         lr=0.02,
                         warmup=args.warmup_proportion,
                         t_total=t_total
                         )

    global_step = 0
    model.train()
    label_list = processor.get_labels(args.task_type)
    label_list_map = dict(zip([i for i in range(len(label_list))], label_list))
    train_steps = len(train_dataloader)
    for e_ in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step in range(train_steps):
            batch = train_iter.next()
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids,output_mask = batch
            # loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            bert_encode = model(input_ids, segment_ids, input_mask)
            loss = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            crf_param_optimizer.step()
            crf_param_optimizer.zero_grad()
            global_step += 1
            if global_step%8==0:
                if args.do_valid:
                    model.eval()
                    with torch.no_grad():
                        # losses = []
                        losses=0.
                        valid_size = 0
                        preds = None
                        out_label_ids = None
                        all_mask = []
                        for step, batch in enumerate(valid_dataloader):
                            batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
                            input_ids, segment_ids, input_mask, label_ids,output_mask = batch
                            # loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
                            predicts =  model.predict_test(bert_encode, output_mask)
                            
                            all_mask.append(output_mask)
                            logits = [[i for i in l.detach().cpu().numpy()] for l in predicts]
                            if preds is None:
                                if type(logits) == list:
                                    preds = logits
                                else:
                                    preds = logits.detach().cpu().numpy()
                                out_label_ids = label_ids.detach().cpu().numpy()
                            else:
                                if type(logits) == list:
                                    preds = np.append(preds, np.asarray(logits), axis=0)
                                else:
                                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                                out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)


                        out_label_ids = out_label_ids.tolist()
                        preds = preds.tolist()

                        all_mask = torch.cat(all_mask, dim=0)
                        all_mask = all_mask.tolist()

                        new_label_ids, new_preds = [], []  
                        for i in range(len(all_mask)):
                            l = sum(all_mask[i])
                            new_preds.append(preds[i][:l])
                            new_label_ids.append(out_label_ids[i][:l])
                        new_label_ids = [t for t in new_label_ids]
                        new_preds = [t for t in new_preds]
                        preds, out_label_ids = new_preds, new_label_ids

                        
                        # assert len(preds) == len(eval_examples)
                        recs = {}
                        for qx, ex in enumerate(valid_examples):
                            recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a, "idx_map": ex.idx_map,
                                                                    "logit": preds[qx]}

                        raw_X = [recs[qx]["sentence"] for qx in range(len(valid_examples)) if qx in recs]
                        idx_map = [recs[qx]["idx_map"] for qx in range(len(valid_examples)) if qx in recs]

                        tokens_list = []
                        for text_a in raw_X:
                            tokens_a = []
                            for t in [token.lower() for token in text_a]:
                                tokens_a.extend(tokenizer.wordpiece_tokenizer.tokenize(t))
                            tokens_list.append(tokens_a[:args.max_seq_length-2])
                        # print(tokens_list[2])    

                        pre = [' '.join([label_list_map.get(p, '-1') for p in l[:args.max_seq_length-2] if p!=-1] ) for l in preds]

                        true = [' '.join([label_list_map.get(p, '-1') for p in l[:args.max_seq_length-2]]) for l in out_label_ids]

                        for i in range(len(true)):
                            assert len(tokens_list[i]) == len(true[i].split()), print(len(tokens_list[i]), len(true[i].split()), tokens_list[i], true[i])
                        lines = [' '.join([str(t) for t in tokens_list[i]]) + '***' + pre[i] + '***' + true[i] for i in range(len(pre))]
                        with open(os.path.join(args.output_dir, 'pre.txt'), 'w') as fp:
                            fp.write('\n'.join(lines))

                        # logger.info('Input data dir: {}'.format(args.data_dir))
                        # logger.info('Output dir: {}'.format(args.output_dir))
                        if args.task_type == 'ae':
                            eval_result(args.output_dir)
                        else:
                            ae,oe=print_valid_result(args.output_dir)
                        logger.info('new ae\oe result: %f,%f',ae,oe)
                        if ae+oe>=max_ae:
                            logger.info('new max ae\oe result: %f,%f',ae,oe)
                            torch.save(model, os.path.join(args.output_dir, "model.pt"))
                            max_ae=ae+oe
                            # max_ae = ae
                        model.train()
        logger.info("epoch: %f", e_)
    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        torch.save(model, os.path.join(args.output_dir, "model.pt"))


def test(args):  # Load a trained model that you have fine-tuned
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels(args.task_type)
    label_list_map = dict(zip([i for i in range(len(label_list))], label_list))
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    eval_examples = processor.get_test_examples_generated_filter(args.data_dir,args.domain_pair,args.dataset_split, args.task_type)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_output_mask)

    # Run prediction for full data and get a prediction file
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = torch.load(os.path.join(args.output_dir, "model.pt"))
    model.cuda()
    model.eval()

    preds = None
    out_label_ids = None
    all_mask = []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, segment_ids, input_mask, label_ids, output_mask = batch

        with torch.no_grad():
            # logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            bert_encode = model(input_ids, segment_ids, input_mask).cpu()
            predicts =  model.predict_test(bert_encode, output_mask)
            
        all_mask.append(output_mask)
        logits = [[i for i in l.detach().cpu().numpy()] for l in predicts]
        if preds is None:
            if type(logits) == list:
                preds = logits
            else:
                preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            if type(logits) == list:
                preds = np.append(preds, np.asarray(logits), axis=0)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)


    out_label_ids = out_label_ids.tolist()
    preds = preds.tolist()

    all_mask = torch.cat(all_mask, dim=0)
    all_mask = all_mask.tolist()

    new_label_ids, new_preds = [], []  
    for i in range(len(all_mask)):
        l = sum(all_mask[i])
        new_preds.append(preds[i][:l])
        new_label_ids.append(out_label_ids[i][:l])
    new_label_ids = [t for t in new_label_ids]
    new_preds = [t for t in new_preds]
    preds, out_label_ids = new_preds, new_label_ids

    output_eval_json = os.path.join(args.output_dir, "predictions.json")
    # assert len(preds) == len(eval_examples)
    recs = {}
    for qx, ex in enumerate(eval_examples):
        recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a, "idx_map": ex.idx_map,
                                                "logit": preds[qx]}

    raw_X = [recs[qx]["sentence"] for qx in range(len(eval_examples)) if qx in recs]
    idx_map = [recs[qx]["idx_map"] for qx in range(len(eval_examples)) if qx in recs]

    tokens_list = []
    for text_a in raw_X:
        tokens_a = []
        for t in [token.lower() for token in text_a]:
            tokens_a.extend(tokenizer.wordpiece_tokenizer.tokenize(t))
        tokens_list.append(tokens_a[:args.max_seq_length-2])
    # print(tokens_list[2])    

    pre = [' '.join([label_list_map.get(p, '-1') for p in l[:args.max_seq_length-2] if p!=-1] ) for l in preds]

    true = [' '.join([label_list_map.get(p, '-1') for p in l[:args.max_seq_length-2]]) for l in out_label_ids]

    for i in range(len(true)):
        assert len(tokens_list[i]) == len(true[i].split()), print(len(tokens_list[i]), len(true[i].split()), tokens_list[i], true[i])
    lines = [' '.join([str(t) for t in tokens_list[i]]) + '***' + pre[i] + '***' + true[i] for i in range(len(pre))]
    with open(os.path.join(args.output_dir, 'pre.txt'), 'w') as fp:
        fp.write('\n'.join(lines))

    logger.info('Input data dir: {}'.format(args.data_dir))
    logger.info('Output dir: {}'.format(args.output_dir))
    if args.task_type == 'ae':
        eval_result(args.output_dir)
    else:
        eval_ts_result(args.output_dir)



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert_e', type=str)

    parser.add_argument("--data_dir",
                        default='service-device',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")

    parser.add_argument('--dataset_split',
                        default='1',
                        type=str,
                        help="random seed for initialization")

    parser.add_argument('--domain_pair',
                        default='rest-device',
                        type=str,
                        help="random seed for initialization")

    parser.add_argument('--task_type',
                        default='aeoe',
                        type=str,
                        help="random seed for initialization")

    parser.add_argument("--output_dir",
                        default='run_out/tmp/service-device',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size", 
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5, 
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=2, 
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()

    # args.output_dir = args.output_dir + str(args.train_batch_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)


if __name__ == "__main__":
    main()

