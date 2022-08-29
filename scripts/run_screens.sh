#!/bin/bash
encoder_name="simclr_x1"
dataset_name="imagenet"

attribution_name=${1}
device1=${2}
device2=${3}
device3=${4}
device4=${5}

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
    command+="source /homes/gws/clin25/miniconda3/etc/profile.d/conda.sh;"
    command+=" conda activate cl-explain-env;"
    command+=" python scripts/run.py"
    command+=" ${encoder_name}"
    command+=" ${explanations[i]}"
    command+=" ${attribution_name}"
    command+=" --dataset-name ${dataset_name}"
    command+=" --batch-size ${batch_size}"
    command+=" --use-gpu"
    command+=" --gpu-num ${devices[i]}"
    command+=" --superpixel-dim ${superpixel_dim}"
    command+=" --eval-superpixel-dim ${eval_superpixel_dim}"
    screen \
    -dmS "${encoder_name}_${explanations[i]}_${attribution_name}_${dataset_name}" \
    bash -c "${command}"
done
