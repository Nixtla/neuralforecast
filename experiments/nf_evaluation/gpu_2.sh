#!/bin/bash

datasets=("M4-quarterly" "M4-monthly" "M4-daily" "M4-hourly")

models=("AutoLSTM" "AutoRNN" "AutoGRU" "AutoDilatedRNN" "AutoDeepAR" "AutoTCN" "AutoMLP" "AutoNBEATS" "AutoNHITS" "AutoDLinear" "AutoTFT" "AutoVanillaTransformer" "AutoInformer" "AutoAutoformer" "AutoFEDformer" "AutoTimesNet" "AutoPatchTST" "AutoTSMixer", "AutoiTransformer")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python run_experiments.py --dataset "$dataset" --model "$model"
    done
done