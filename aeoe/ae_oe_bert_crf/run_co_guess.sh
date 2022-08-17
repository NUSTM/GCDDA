#!/bin/bash
domains=('restaurants' 'laptops' 'device')
export CUDA_VISIBLE_DEVICES=0
# output='./run_out/ds-bert-e-32/'
# output2='./run_out/ds-bert-e-42/'
output1='./run_out/source-52-small-100/'
# domains1=('restaurants')
# domains=('device')

output2='./dp_data/'
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
            echo "${src_domain}-${tar_domain}"
	        python ./co-guess.py \
                --data_dir1 "${output1}#${src_domain}-${tar_domain}"  \
                --data_dir2 "${output2}#${src_domain}-${tar_domain}"  \
                --output_dir "./baseline-filter-3-new-mask-0.6-0.6-10-sample-final-small-100/#${src_domain}_${tar_domain}"
        fi
    done
done



