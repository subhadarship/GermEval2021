import os

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import load_dataframe

if __name__ == "__main__":
    """Train and dev data"""
    INP_TRAIN_DATA_PATH = os.path.join('../data/GermEval21_Toxic_Train/GermEval21_Toxic_Train.csv')
    OUT_TRAIN_DIR = os.path.join('../data/GermEval21_Toxic_Train/')
    assert os.path.isfile(INP_TRAIN_DATA_PATH)
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)

    # load data
    raw_train_data_df = load_dataframe(INP_TRAIN_DATA_PATH)

    # split into train and dev
    train_df, dev_df = train_test_split(raw_train_data_df, test_size=0.2,
                                        stratify=raw_train_data_df['Sub3_FactClaiming'],
                                        random_state=123)

    # write to files
    train_df.to_csv(os.path.join(OUT_TRAIN_DIR, 'train.csv'), encoding='utf-8', index=False)
    dev_df.to_csv(os.path.join(OUT_TRAIN_DIR, 'dev.csv'), encoding='utf-8', index=False)

    """Test data"""
    INP_TEST_DATA_PATH = os.path.join('../data/GermEval21_Toxic_Test/GermEval21_Toxic_TestData.csv')
    OUT_TEST_DIR = os.path.join('../data/GermEval21_Toxic_Test')
    assert os.path.isfile(INP_TEST_DATA_PATH)
    os.makedirs(OUT_TEST_DIR, exist_ok=True)

    raw_test_data_df = load_dataframe(INP_TEST_DATA_PATH)
    raw_test_data_df.rename(columns={'c_text': 'comment_text'}, inplace=True)
    raw_test_data_df.to_csv(os.path.join(OUT_TEST_DIR, 'test.csv'), encoding='utf-8', index=False)

    """5-fold cross validation data"""
    CROSS_VALIDATION_DATA_DIR = os.path.join('../data/cross_validation')

    # split into train and dev
    part1, part2 = train_test_split(train_df, test_size=0.5,
                                    stratify=train_df['Sub3_FactClaiming'],
                                    random_state=123)
    part1a, part1b = train_test_split(part1, test_size=0.5,
                                      stratify=part1['Sub3_FactClaiming'],
                                      random_state=123)
    part2a, part2b = train_test_split(part2, test_size=0.5,
                                      stratify=part2['Sub3_FactClaiming'],
                                      random_state=123)

    part0 = dev_df.copy()

    train_dfs_dict = {
        ('fold_A', 'dev'): part0,
        ('fold_B', 'dev'): part1a,
        ('fold_C', 'dev'): part1b,
        ('fold_D', 'dev'): part2a,
        ('fold_E', 'dev'): part2b,
        ('fold_A', 'train'): train_df.copy(),
        ('fold_B', 'train'): pd.concat([part0, part1b, part2a, part2b]),
        ('fold_C', 'train'): pd.concat([part0, part1a, part2a, part2b]),
        ('fold_D', 'train'): pd.concat([part0, part1a, part1b, part2b]),
        ('fold_E', 'train'): pd.concat([part0, part1a, part1b, part2a]),
    }

    for (fold_name, split_name), dataframe in train_dfs_dict.items():
        os.makedirs(os.path.join(CROSS_VALIDATION_DATA_DIR, fold_name), exist_ok=True)
        dataframe.to_csv(os.path.join(CROSS_VALIDATION_DATA_DIR, fold_name, f'{split_name}.csv'), encoding='utf-8',
                         index=False)
