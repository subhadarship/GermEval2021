import pandas as pd
import os

if __name__ == "__main__":
    LABEL_COLUMN_NAMES = ['Sub1_Toxic', 'Sub2_Engaging', 'Sub3_FactClaiming']
    PRED_DIR = os.path.join('../predictions/best_models/')
    SUBMISSION_DIR = os.path.join('../submission')
    TEST_INP_FILE_PATH = os.path.join('../data/GermEval21_Toxic_Test/GermEval21_Toxic_TestData.csv')
    test_df = pd.read_csv(TEST_INP_FILE_PATH, encoding='utf-8')[['comment_id']]

    for folder in os.listdir(PRED_DIR):
        preds = \
            pd.read_csv(os.path.join(PRED_DIR, folder, f'test.Sub3_FactClaiming.out'), encoding='utf-8', header=None)[
                0].to_list()
        submission_df = test_df.copy(deep=True)
        submission_df[LABEL_COLUMN_NAMES[0]] = [0] * len(submission_df)
        submission_df[LABEL_COLUMN_NAMES[1]] = [0] * len(submission_df)
        submission_df[LABEL_COLUMN_NAMES[2]] = preds
        os.makedirs(os.path.join(SUBMISSION_DIR, folder), exist_ok=True)
        submission_df.to_csv(os.path.join(SUBMISSION_DIR, folder, 'answer.csv'), encoding='utf-8', index=False)
