#!/bin/bash

encoder_name=${1}
dataset_name=${2}
attribution_name=${3}

device1=${4}
device2=${5}
device3=${6}
device4=${7}
shift 7
while getopts "nd" opt
do
    case $opt in
        n)  normalize_similarity=true;;
        d)  different_classes=true;;
        *)  exit 1;;
    esac
done

devices=( "${device1}" "${device2}" "${device3}" "${device4}" )
explanations=(
    "self_weighted"
    "contrastive_self_weighted"
    "corpus"
    "contrastive_corpus"
)

if [ "${attribution_name}" = "int_grad" ] \
    || [ "${attribution_name}" = "smooth_int_grad" ] \
    || [ "${attribution_name}" = "gradient_shap" ]
then
    batch_size=1
elif [ "${attribution_name}" = "smooth_vanilla_grad" ]
then
    batch_size=4
else
    batch_size=32
fi

# No need for superpixels when explaining cifar
if [ "${attribution_name}" = "kernel_shap" ] \
    && [ "${dataset_name}" != "cifar" ]
then
    superpixel_dim=8
    eval_superpixel_dim=8
else
    superpixel_dim=1
    eval_superpixel_dim=1
fi

for i in {0..3}
do
    command=""
    screen_name=""

    command+="source /homes/gws/clin25/miniconda3/etc/profile.d/conda.sh;"
    command+=" conda activate cl-explain-env;"
    command+=" python scripts/run.py"
    command+=" ${encoder_name}"
    command+=" ${explanations[i]}"
    command+=" ${attribution_name}"

    if [ "${normalize_similarity}" = true ]
    then
        command+=" --normalize-similarity"
        screen_name+="norm_"
    else
        screen_name+="unnorm_"
    fi

    if [ "${different_classes}" = true ]
    then
        command+=" --different-classes"
        screen_name+="diff_"
    else
        screen_name+="same_"
    fi

    command+=" --dataset-name ${dataset_name}"
    command+=" --batch-size ${batch_size}"
    command+=" --use-gpu"
    command+=" --gpu-num ${devices[i]}"
    command+=" --superpixel-dim ${superpixel_dim}"
    command+=" --eval-superpixel-dim ${eval_superpixel_dim}"

    screen_name+="${encoder_name}_${explanations[i]}"
    screen_name+="_${attribution_name}_${dataset_name}"

    screen -dmS "${screen_name}" bash -c "${command}"
done
