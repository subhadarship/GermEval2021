#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1

MODELDIR=../models/best_models
LOGDIR=../logs_best_models
HIDDIM=512
LR=0.0005
FOLD=fold_A
# BERT parameters are frozen

# Monotask multilingual BERT
MODELNAME=bert-base-multilingual-cased
expt=monotask_${MODELNAME}
CUDA_VISIBLE_DEVICES=${GPUID} python train.py --label_col_names Sub3_FactClaiming --eval_label_col_names Sub3_FactClaiming --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert --train_data_dir ../data/cross_validation/${FOLD} --dev_data_dir ../data/cross_validation/${FOLD}

# Monotask German BERT
MODELNAME=bert-base-german-cased
expt=monotask_${MODELNAME}
CUDA_VISIBLE_DEVICES=${GPUID} python train.py --label_col_names Sub3_FactClaiming --eval_label_col_names Sub3_FactClaiming --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert --train_data_dir ../data/cross_validation/${FOLD} --dev_data_dir ../data/cross_validation/${FOLD}

# Multitask German BERT
MODELNAME=bert-base-german-cased
expt=multitask_${MODELNAME}
CUDA_VISIBLE_DEVICES=${GPUID} python train.py --label_col_names Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming --eval_label_col_names Sub3_FactClaiming --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert --train_data_dir ../data/cross_validation/${FOLD} --dev_data_dir ../data/cross_validation/${FOLD}
