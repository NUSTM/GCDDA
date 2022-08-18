#!/bin/bash

domains=('restaurants' 'laptops' 'device')
export CUDA_VISIBLE_DEVICES=0

for split in `seq 1 3`
do
output="../../run_out/source_${split}/"

for src_domain in ${domains[@]};
do
    for tar_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then   
            echo "${src_domain}-${tar_domain}"
  	        python ./run_bert_absa_da_filter.py --task_type 'aeoe' \
                  --data_dir "../../da/da_data${split}/" \
                  --domain_pair "${src_domain}-${tar_domain}" \
                  --dataset_split ${split} \
                  --output_dir "${output}${src_domain}-${tar_domain}"  \
                  --train_batch_size 16 \
                  --bert_model 'bert_e' \
                  --seed 52 \
                  --do_eval
            echo "${src_domain}-${tar_domain}"
	        python ./co-guess.py \
                --data_dir "${output}#${src_domain}-${tar_domain}"  \
                --output_dir "./filter_da${split}/#${src_domain}_${tar_domain}"
        fi
        
    done
done
done





