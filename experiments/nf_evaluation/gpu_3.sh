#!/bin/bash

datasets=("Ettm2" "Ettm1" "Etth1" "Etth2")

models=("AutoLSTM" "AutoRNN" "AutoGRU" "AutoDilatedRNN" "AutoDeepAR" "AutoTCN" "AutoMLP" "AutoNBEATS" "AutoNHITS" "AutoDLinear" "AutoTFT" "AutoVanillaTransformer" "AutoInformer" "AutoAutoformer" "AutoFEDformer" "AutoTimesNet" "AutoPatchTST")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=2 python run_experiments.py --dataset "$dataset" --model "$model"
    done
done