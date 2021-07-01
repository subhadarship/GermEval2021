from argparse import ArgumentParser


def get_predict_args():
    """Prediction args for GermEval classification model"""
    parser = ArgumentParser(description='prediction using GermEval classification model')
    parser.add_argument('--log_file_path', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='../models/tmp')
    parser.add_argument('--model_name', type=str, default='transformer_enc')
    parser.add_argument('--label_col_names', type=str, default='Sub3_FactClaiming')
    parser.add_argument('--predict_label_col_names', type=str, default='Sub3_FactClaiming')
    parser.add_argument('--dev_path', type=str, default=None)
    parser.add_argument('--test_inp_path', type=str, default=None)
    parser.add_argument('--test_pred_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4096)
    args = parser.parse_args()

    return args
