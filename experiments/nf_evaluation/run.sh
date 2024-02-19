#!/bin/bash

datasets=("M3-yearly" "M3-quarterly" "M3-monthly" "M4-yearly" "M4-quarterly" "M4-monthly" "M4-daily" "M4-hourly" "Ettm2" "Ettm1" "Etth1" "Etth2" "Electricity" "Exchange" "Weather" "Traffic" "ILI")

models=("AutoLSTM" "AutoRNN" "AutoGRU" "AutoDilatedRNN" "AutoDeepAR" "AutoTCN" "AutoMLP" "AutoNBEATS" "AutoNHITS" "AutoDLinear" "AutoTFT" "AutoVanillaTransformer" "AutoInformer" "AutoAutoformer" "AutoFEDformer" "AutoTimesNet" "AutoPatchTST")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python run_experiments.py --dataset "$dataset" --model "$model"
    done
done