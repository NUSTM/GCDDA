import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch

class Bert_CRF(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_tag):
        super(Bert_CRF, self).__init__(config)
        self.bert = BertModel(config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
        self.apply(self.init_bert_weights)

        self.crf = CRF(num_tag)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None,
                output_all_encoded_layers=False):
        bert_encode, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=output_all_encoded_layers)
        
        output = self.classifier(bert_encode)
        return output

    def loss_fn(self, bert_encode, output_mask, tags):
        
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def predict_test(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)

    # def print_test_entity(self, preds,test_examples,all_mask,synonyms):
    #     from difflib import SequenceMatcher#导入库
    #     def similarity(a, b):
    #         return SequenceMatcher(None, a, b).ratio()
    #     label_list_map={0:"B-EN", 1:"I-EN", 2:"O"}
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    #     preds = preds.tolist()
    #     all_mask = torch.cat(all_mask, dim=0)
    #     all_mask = all_mask.tolist()

    #     # get rid of paddings and sepacial tokens([CLS] and [SEP])
    #     new_label_ids, new_preds = [], []  
    #     for i in range(len(all_mask)):
    #         l = sum(all_mask[i])
    #         new_preds.append(preds[i][:l])
    #     new_preds = [t for t in new_preds]
    #     preds = new_preds

    #     output_eval_json = os.path.join(args.output_dir, "predictions.json")
        
    #     # assert len(preds) == len(eval_examples)
    #     recs = {}
    #     for qx, ex in enumerate(test_examples):
    #         recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a,
    #                                             "logit": preds[qx]}

    #     raw_X = [recs[qx]["sentence"] for qx in range(len(test_examples)) if qx in recs]


    #     tokens_list = []
    #     for text_a in raw_X:
    #         tokens_a = []
            
    #         tokens_a=[token.lower() for token in text_a.split()]
    #         #     tokens_a.extend(tokenizer._convert_token_to_id(t))
            
    #         tokens_list.append(tokens_a[:args.max_seq_length-2])
    #     # print(tokens_list[2])    

    #     pre = [' '.join([label_list_map.get(p, '-1') for p in l[:args.max_seq_length-2] if p!=-1] ) for l in preds]
    #     match_pre=[]
    #     specil_match=[]
    #     specil_index=[]
    #     for j,label in enumerate(pre):
    #         max_sorce=-1
    #         max_entity=''
    #         en_index=[]
    #         for i,l in enumerate(label.split()):
    #             if l !='O':
    #                 en_index.append(i)
     
    #         if en_index==[]:
    #             entity=''
    #         else:
    #             entity=''.join(tokens_list[j][en_index[0]:en_index[-1]+1])
    #         # print(entity)
    #         if entity in synonyms:
    #             match_pre.append(synonyms[entity])
    #         else:
    #             for s in synonyms.keys():
    #                 score=similarity(''.join([str(t) for t in tokens_list[j]]), s)
    #                 if score>max_sorce:
    #                     max_entity=synonyms[s]
    #                     max_sorce=score
                
    #             match_pre.append(max_entity)
    #             specil_match.append(max_entity)
    #             specil_index.append(j)
                
                    
    #     lines = [''.join(tokens_list[i]) + '***' + pre[i] for i in range(len(pre)) ]
        
    #     lines_match= [''.join(tokens_list[i]) + '***' + match_pre[i] for i in range(len(match_pre))]

    #     lines_specil= [''.join(tokens_list[specil_index[i]]) + '***' + specil_match[i] for i in range(len(specil_index))]
    #     with open(os.path.join(args.output_dir, 'pre.txt'), 'w') as fp:
    #         fp.write('\n'.join(lines))
    #     with open(os.path.join(args.output_dir, 'match_pre.txt'), 'w') as fp:
    #         fp.write('\n'.join(lines_match))
    #     with open(os.path.join(args.output_dir, 'specil_pre.txt'), 'w') as fp:
    #         fp.write('\n'.join(lines_specil))



