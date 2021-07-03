import os

from training_utils import compute_metrics
from typing import List


def load_labels(fpath: str) -> List[str]:
    """Load labels from file"""
    assert os.path.isfile(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = list(map(str.strip, f.readlines()))
    labels = list(map(lambda x: f'{x}', lines))
    return labels


if __name__ == "__main__":
    REF_PATH = os.path.join('../references/dev.Sub3_FactClaiming.ref.tsv')
    PRED_PATH = os.path.join('../predictions/best_models/monotask_bert-base-german-cased/dev.Sub3_FactClaiming.out')

    gold_labels = load_labels(REF_PATH)
    pred_labels = load_labels(PRED_PATH)

    print(
        compute_metrics(
            gold_labels=[[gold_label] for gold_label in gold_labels],
            predictions=[[pred_label] for pred_label in pred_labels],
            all_classes=['0', '1'],
            desired_label_ids=[0],
        )
    )
