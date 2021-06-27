import os


def add_data_args(parser):
    """Args related to data"""
    parser.add_argument('--train_data_dir', type=str,
                        default=os.path.join('../data'))
    parser.add_argument('--dev_data_dir', type=str,
                        default=os.path.join('../data'))
    parser.add_argument('--test_data_dir', type=str,
                        default=None)
    parser.add_argument('--label_col_names', type=str, default='Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_vocab_size', type=int, default=None)  # not applicable when model is BERT based
    parser.add_argument('--tokenization', type=str, default='tweet')
    return parser
