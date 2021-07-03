import os

from data_utils import load_dataframe

if __name__ == "__main__":
    DATA_DIR = '../data/GermEval21_Toxic_Train'
    REFERENCES_DIR = '../references'
    LABEL_COL_NAME = 'Sub3_FactClaiming'

    os.makedirs(REFERENCES_DIR, exist_ok=True)

    # dev data
    df = load_dataframe(os.path.join(DATA_DIR, f'dev.csv'))
    df = df.filter([LABEL_COL_NAME])
    df.to_csv(
        os.path.join(REFERENCES_DIR, f'dev.{LABEL_COL_NAME}.ref.tsv'),
        sep='\t', encoding='utf-8', index=False, header=False
    )
