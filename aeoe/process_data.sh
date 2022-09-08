#!/bin/bash
domains=('restaurants' 'laptops' 'device')
# domains=('laptop')
# domains2=('rest')
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            
            
            python ungram.py \
                --source_domain ${src_domain} \
                --target_domain ${tar_domain} \
                --dataset_split 3
            
           
        fi
    done
done
