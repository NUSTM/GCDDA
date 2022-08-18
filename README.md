## GCDDA

### environment
transformers==4.2.2  
pytorch==1.10.0  
### code
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
