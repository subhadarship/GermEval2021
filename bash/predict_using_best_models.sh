#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1
MODELDIR=../models/best_models
DEVPATH=../data/cross_validation/fold_A/dev.csv
TESTINPPATH=../data/GermEval21_Toxic_Test/test.csv
PREDPATH=../predictions/best_models
DEVPREDNAME=dev.Sub3_FactClaiming.out
TESTPREDNAME=test.Sub3_FactClaiming.out

# Monotask multilingual BERT
MODELNAME=bert-base-multilingual-cased
expt=monotask_${MODELNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --test_inp_path ${DEVPATH} --test_pred_path ${PREDPATH}/${expt}/${DEVPREDNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --dev_path ${DEVPATH} --test_inp_path ${TESTINPPATH} --test_pred_path ${PREDPATH}/${expt}/${TESTPREDNAME}

# Monotask German BERT
MODELNAME=bert-base-german-cased
expt=monotask_${MODELNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --test_inp_path ${DEVPATH} --test_pred_path ${PREDPATH}/${expt}/${DEVPREDNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --dev_path ${DEVPATH} --test_inp_path ${TESTINPPATH} --test_pred_path ${PREDPATH}/${expt}/${TESTPREDNAME}

# Multitask German BERT
MODELNAME=bert-base-german-cased
expt=multitask_${MODELNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --test_inp_path ${DEVPATH} --test_pred_path ${PREDPATH}/${expt}/${DEVPREDNAME}
python predict.py --model_dir ${MODELDIR}/${expt} --model_name ${MODELNAME} --label_col_names Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming --predict_label_col_names Sub3_FactClaiming --dev_path ${DEVPATH} --test_inp_path ${TESTINPPATH} --test_pred_path ${PREDPATH}/${expt}/${TESTPREDNAME}
