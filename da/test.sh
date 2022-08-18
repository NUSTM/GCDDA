#!/bin/bash
domains=('restaurants' 'laptops' 'device')
# domains=('laptops')
#domains2=('restaurants')
export CUDA_VISIBLE_DEVICES=0
for dataset_split in `seq 1 3`;
do
    for src_domain in ${domains[@]};
    do
        for tar_domain in  ${domains[@]};
        do
            # if [ $src_domain != 'device' ];
            # then
            #      continue
            # fi
            if [ $src_domain != $tar_domain ];
            then
                echo "${src_domain}-${tar_domain}"
                python run-strict.py \
                    --model_name_or_path facebook/bart-base \
                    --source_file ../aeoe/masked_source_label/${src_domain}_${tar_domain}_${dataset_split}_lin.txt \
                    --target_file ../aeoe/masked_target_pseudo_label/${src_domain}_${tar_domain}_${dataset_split}_lin.txt \
                    --source_raw_file ../aeoe/source_label/${src_domain}_${tar_domain}_${dataset_split}_lin.txt \
                    --target_raw_file ../aeoe/target_pseudo_label/${src_domain}_${tar_domain}_${dataset_split}_lin.txt \
                    --source_file_random ../aeoe/source_label-random/${src_domain}_${tar_domain}_${dataset_split}_lin.txt \
                    --text_column Text \
                    --summary_column Summary \
                    --output_dir output/${src_domain}_${tar_domain}${dataset_split}/ \
                    --overwrite_output_dir \
                    --do_train \
                    --do_eval \
                    --min_summ_length=100 \
                    --max_summ_length=250 \
                    --length_penalty=1.0 \
                    --per_device_train_batch_size=4 \
                    --per_device_eval_batch_size=4 \
                    --predict_with_generate \
                    --seed 42
            fi
        done
    done
done