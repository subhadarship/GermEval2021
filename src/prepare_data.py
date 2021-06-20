import os

from sklearn.model_selection import train_test_split

from data_utils import load_dataframe

if __name__ == "__main__":
    INP_DATA_PATH = os.path.join('../data/GermEval21_Toxic_Train/GermEval21_Toxic_Train.csv')
    OUT_DIR = os.path.join('../data/GermEval21_Toxic_Train/')
    assert os.path.isfile(INP_DATA_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # load data
    raw_data_df = load_dataframe(INP_DATA_PATH)

    # split into train and dev
    train_df, dev_df = train_test_split(raw_data_df, test_size=0.2, stratify=raw_data_df['Sub3_FactClaiming'],
                                        random_state=123)

    # write to files
    train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), encoding='utf-8', index=False)
    dev_df.to_csv(os.path.join(OUT_DIR, 'dev.csv'), encoding='utf-8', index=False)
