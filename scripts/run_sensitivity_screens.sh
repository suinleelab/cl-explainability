#!/bin/bash

encoder_name=${1}
parameter=${2}
attribution_name=${3}

device1=${4}
device2=${5}
device3=${6}
device4=${7}
device5=${8}

shift 8
while getopts "n" opt
do
    case $opt in
        n)  normalize_similarity=true;;
        *)  exit 1;;
    esac
done

devices=( "${device1}" "${device2}" "${device3}" "${device4}" "${device5}" )
seeds=( 123 456 789 42 91 )


for i in {0..4}
do
    device="${devices[i]}"
    seed="${seeds[i]}"

    if [ "${device}" = "-1" ]
    then
        continue
    fi

    command=""
    screen_name="sensitivity_"

    command+="source /homes/gws/clin25/miniconda3/etc/profile.d/conda.sh;"
    command+=" conda activate cl-explain-env;"
    command+=" python scripts/analyze_sensitivity.py"
    command+=" ${encoder_name}"
    command+=" ${parameter}"
    command+=" --attribution-name ${attribution_name}"

    if [ "${normalize_similarity}" = true ]
    then
        command+=" --normalize-similarity"
        screen_name+="norm_"
    else
        screen_name+="unnorm_"
    fi

    command+=" --gpu-num ${device}"
    command+=" --seed ${seed}"

    screen_name+="${encoder_name}_${parameter}_${attribution_name}"
    screen_name+="_${seed}"

    screen -dmS "${screen_name}" bash -c "${command}"
done
