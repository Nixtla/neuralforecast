#!/bin/bash

datasets=("M3-yearly" "M3-quarterly" "M3-monthly" "M4-yearly")

models=("AutoLSTM" "AutoRNN" "AutoGRU" "AutoDilatedRNN" "AutoDeepAR" "AutoTCN" "AutoMLP" "AutoNBEATS" "AutoNHITS" "AutoDLinear" "AutoTFT" "AutoVanillaTransformer" "AutoInformer" "AutoAutoformer" "AutoFEDformer" "AutoTimesNet" "AutoPatchTST", "AutoTSMixer", "AutoiTransformer")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python run_experiments.py --dataset "$dataset" --model "$model"
    done
done