import os

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
