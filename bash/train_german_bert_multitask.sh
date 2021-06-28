#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1

# German BERT

MODELNAME=bert-base-german-cased

MODELDIR=/mnt/backup/panda/GermEval2021/models

# BERT parameters are trainable
for HIDDIM in 128 256 512; do
  for LR in 0.0005 0.005 0.05; do

    expt=multitask/${MODELNAME}/trg_fc${HIDDIM}_lr${LR}_trainable
    CUDA_VISIBLE_DEVICES=${GPUID} python train.py --label_col_names Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming --eval_label_col_names Sub3_FactClaiming --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
    rm -rf ${MODELDIR}/${expt}

  done
done

# BERT parameters are frozen
for HIDDIM in 128 256 512; do
  for LR in 0.0005 0.005 0.05; do

    expt=multitask/${MODELNAME}/trg_fc${HIDDIM}_lr${LR}_frozen
    CUDA_VISIBLE_DEVICES=${GPUID} python train.py --label_col_names Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming --eval_label_col_names Sub3_FactClaiming --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
    rm -rf ${MODELDIR}/${expt}

  done
done
