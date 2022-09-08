
# Generative Cross-Domain Data Augmentation for Aspect and Opinion Co-Extraction
This repository contains code for our NAACL2022 paper:  
[Generative Cross-Domain Data Augmentation for Aspect and Opinion Co-Extraction](https://aclanthology.org/2022.naacl-main.312.pdf)
## Datasets

The training data comes from three domains: Restaurant(R) 、 Laptop(L) 、 Device(D).  
We follow the previous work and remove the sentences that have no aspects and opinions when device is the source domain.  

The in-domain corpus(used for training BERT-E) come from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 

Click here to get [BERT-E](https://pan.baidu.com/s/1hNyNCyfOHzznuPbxT1LNFQ) (BERT-Extented) , and the extraction code is by0i. (Please specify the directory where BERT is stored in modelconfig.py.)

## environment
transformers==4.2.2  
pytorch==1.10.0  
## code
1. Firstly, we run the following code to achieve the target pseudo labeled data:
```
cd aeoe
cd ae_oe_bert_crf
bash ./run_bert_e_sdl.sh
```
2. Then, we run the following code to achieve masked data:
```
cd ..
bash ./process_data.sh
```
3. After that, we train the bart for data generation:
```
cd ..
cd da
bash ./test.sh
bash ./post_process.sh
```
4. finally, we filter the generated data and train it for downstreamtask:
```
cd ..
cd aeoe
cd ae_oe_bert_crf
bash ./run_bert_e_da_filter.sh
bash ./run_co_guess.sh
bash ./run_bert_e_da_train.sh
```
