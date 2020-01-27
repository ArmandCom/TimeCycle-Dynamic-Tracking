#!/bin/bash

for name in checkpoint_14.pth.tar
do
YOUR_DATASET_FOLDER='/data/Armand/TimeCycle'

echo $name
if [ ! -d "results/davis_results__$name/" ]; then
    echo $name
    python test_davis.py \
    --evaluate --cropSize 320 --cropSize2 80 --gpu-id 0,1,2,3 --topk_vis 5 \
    --resume pytorch_checkpoints/release_model_simple/$name \
    --save_path results/davis_results__$name/
fi

python2 preprocess/convert_davis.py --in_folder results/davis_results__$name/ --out_folder results/davis_results__${name}_converted/ --dataset $YOUR_DATASET_FOLDER/davis/ && \
python2 $YOUR_DATASET_FOLDER/davis-2017/python/tools/eval.py -i results/davis_results__${name}_converted/ -o results/davis_results__${name}_converted/results.yaml --year 2017 --phase val \
| tee results/davis_results__${name}_converted/output.txt &

done