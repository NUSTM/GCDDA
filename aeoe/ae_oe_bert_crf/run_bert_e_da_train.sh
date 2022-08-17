#!/bin/bash
domains=('restaurants' 'laptops' 'device')
export CUDA_VISIBLE_DEVICES=0
output='../../run_out/'
for split in '3'
do
for src_domain in ${domains[@]};
do
    for tar_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then   
            echo "${src_domain}-${tar_domain}"
  	        python ./run_bert_absa_da_filter_train.py --task_type 'aeoe' \
                  --data_dir "./filter_da${split}/" \
                  --domain_pair "${src_domain}-${tar_domain}" \
                  --dataset_split ${split} \
                  --output_dir "${output}${src_domain}-${tar_domain}"  \
                  --train_batch_size 16 \
                  --bert_model 'bert_e' \
                  --seed 52 \
                  --do_train \
                  --do_eval
        fi
    done
done
done





