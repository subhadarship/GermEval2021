import logging
import os
from typing import Union, List

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from args_utils import get_predict_args
from data_utils import (
    load_dataframe,
    GermEvalDataset,
    BertGermEvalDataset,
    SMARTTOKDataLoader,
    LabelField,
)
from model_utils import (
    load_transformer_enc_multitask_classification_model,
    load_bert_multitask_classification_model,
    load_logistic_regression_multitask_classification_model,
    load_checkpoint,
    MultitaskBertClassificationModel,
    MultitaskLogisticRegressionClassificationModel,
    MultitaskTransformerEncoderClassificationModel,
)
from training_utils import (
    init_logger,
    evaluate,
)
from training_utils import postprocess_labels

logger = logging.getLogger(__name__)


def predict(
        _model: Union[
            MultitaskTransformerEncoderClassificationModel,
            MultitaskBertClassificationModel,
            MultitaskLogisticRegressionClassificationModel
        ],
        iterator: Union[SMARTTOKDataLoader],
        label_fields: List[LabelField],
        desired_label_ids: List[int],
) -> List[List[str]]:
    """Predict method"""

    # set model to eval mode
    _model.eval()

    all_preds = []

    test_meter = tqdm(iterator, desc='predict', unit=' batches', leave=False, total=0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_meter):
            test_meter.total = len(iterator)
            text = batch['text']

            # forward pass
            outs = _model(text)

            logits = outs['logits']

            # logits = {L1: [batch size, output dim], ...}

            # compute prediction
            preds = {k: v.max(dim=1)[1] for k, v in logits.items()}

            all_preds.append({k: v.detach().cpu() for k, v in preds.items()})

    # flatten
    flattened_preds = torch.cat(
        [torch.cat([_p[f'L{idx + 1}'].view(-1, 1) for idx in range(len(_p))], dim=1) for _p in all_preds]).tolist()

    # postprocess
    logger.info('postprocessing predictions..')
    flattened_preds_postprocessed = postprocess_labels(flattened_preds, label_fields)

    return [[flat[desired_idx] for desired_idx in desired_label_ids] for flat in flattened_preds_postprocessed]


if __name__ == "__main__":

    # get predict args
    args = get_predict_args()

    # init logger
    init_logger(args.log_file_path)
    logger.info("\n\n*****************\n***RUN STARTED***\n*****************\n")

    # log args
    args_str = f'args\n{89 * "-"}\n'
    for k, v in args.__dict__.items():
        args_str += f'\t{k}: {v}\n'
    args_str += f'{89 * "-"}\n'
    logger.info(args_str)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    """Load data"""
    data_dfs_dict = {
        'dev': load_dataframe(args.dev_path) if args.dev_path is not None else None,
        'test': load_dataframe(args.test_inp_path) if args.test_inp_path is not None else None,
    }

    if data_dfs_dict['test'] is not None:
        # add dummy labels
        for label_name in ['Sub1_Toxic', 'Sub2_Engaging', 'Sub3_FactClaiming']:
            data_dfs_dict['test'][label_name] = [0] * len(data_dfs_dict['test'])

    """Load checkpoint"""
    # load checkpoint
    logger.info(f'load checkpoint from {args.model_dir}')
    checkpoint_dict = load_checkpoint(args.model_dir, device)
    TEXT = checkpoint_dict['data_dict']['TEXT'] if 'TEXT' in checkpoint_dict['data_dict'] else None
    LABELS = checkpoint_dict['data_dict']['LABELS']

    """Preprocess data"""
    datasets_dict, dataloaders_dict = {}, {}
    for split_name in data_dfs_dict:
        if data_dfs_dict[split_name] is None:
            continue
        if args.model_name == 'transformer_enc':
            datasets_dict[split_name] = GermEvalDataset(
                df=data_dfs_dict[split_name],
                label_col_names=args.label_col_names.split(','),
                text_field=TEXT,
                label_fields=LABELS,
                build_vocab=False,
                max_len=1000,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=TEXT.stoi[TEXT.pad_token],
                shuffle=False,
                progress_bar=True,
                device=device
            )
        elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased', 'bert-base-german-cased']:
            datasets_dict[split_name] = BertGermEvalDataset(
                df=data_dfs_dict[split_name],
                label_col_names=args.label_col_names.split(','),
                bert_tokenizer=checkpoint_dict['data_dict']['TOKENIZER'],
                label_fields=LABELS,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=checkpoint_dict['data_dict']['TOKENIZER'].pad_token_id,
                shuffle=False,
                progress_bar=True,
                device=device
            )
        elif args.model_name == 'logistic_regression':
            datasets_dict[split_name] = GermEvalDataset(
                df=data_dfs_dict[split_name],
                label_col_names=args.label_col_names.split(','),
                text_field=TEXT,
                label_fields=LABELS,
                build_vocab=False,
                max_len=None,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=TEXT.stoi[TEXT.pad_token],
                shuffle=False,
                progress_bar=True,
                device=device
            )
        else:
            raise NotImplementedError

    """Load model"""

    if args.model_name == 'transformer_enc':
        model = load_transformer_enc_multitask_classification_model(
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            device=device,
        )
    elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased', 'bert-base-german-cased']:
        model = load_bert_multitask_classification_model(
            model_name=args.model_name,
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            freeze=True,  # setting to True/False does not affect predictions
            device=device,
        )
    elif args.model_name == 'logistic_regression':
        model = load_logistic_regression_multitask_classification_model(
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            device=device,
        )
    else:
        raise NotImplementedError

    # log model
    logger.info(f'model\n{89 * "-"}\n{str(model)}\n{89 * "-"}\n')
    logger.info(
        f'the model has '
        f'{sum(p.numel() for p in model.parameters()):,} '
        f'total parameters (both trainable/non-trainable)'
    )

    # sanity check
    assert len(set(args.label_col_names.split(',')).union(set(args.predict_label_col_names.split(',')))) == len(
        set(args.label_col_names.split(',')))
    # get desired label ids
    desired_label_ids = [args.label_col_names.split(',').index(predict_label_col_name) for predict_label_col_name in
                         args.predict_label_col_names.split(',')]

    """Predict"""
    # criterion
    criterion = nn.CrossEntropyLoss()

    # load model weights
    logger.info(f'load model weights from checkpoint in {args.model_dir}')
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    if 'dev' in dataloaders_dict:
        # compute val loss
        logger.info(f'🔥 start prediction on dev inputs..')
        valid_loss, valid_metrics = evaluate(model=model, iterator=dataloaders_dict['dev'], criterion=criterion,
                                             label_fields=LABELS, all_classes=["0", "1"],
                                             desired_label_ids=desired_label_ids, )
        logger.info(f'val_loss: {valid_loss:.3f}')
        logger.info(f'🔥 validation metrics 🔥 {valid_metrics}')

    if 'test' in dataloaders_dict:
        # predict on test inputs
        logger.info(f'🔥 start prediction on test inputs..')
        test_predictions = predict(_model=model, iterator=dataloaders_dict['test'], label_fields=LABELS,
                                   desired_label_ids=desired_label_ids, )
        if args.test_pred_path is not None:
            if os.path.dirname(args.test_pred_path) != '':
                os.makedirs(os.path.dirname(args.test_pred_path), exist_ok=True)
            logger.info(f'writing test predictions to {args.test_pred_path}..')
            test_predictions_df = pd.DataFrame(test_predictions)
            test_predictions_df.to_csv(
                args.test_pred_path,
                sep='\t', encoding='utf-8', index=False, header=False
            )
