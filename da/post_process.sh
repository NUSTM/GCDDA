#!/bin/bash
domains=('restaurants' 'laptops' 'device')
# domains=('laptop')
# domains2=('rest')
export CUDA_VISIBLE_DEVICES=3
for split in `seq 1 3`
do
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do  
        
        if [ $src_domain != $tar_domain ];
        then
            python post_process.py \
                --source_domain ${src_domain} \
                --target_domain ${tar_domain} \
                --dataset_split ${split}        
        fi
    done
done
done