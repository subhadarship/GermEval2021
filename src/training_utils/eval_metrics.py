from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(
        gold_labels: List[List[str]],
        predictions: List[List[str]],
        all_classes: List[str],
        desired_label_ids: List[int],
) -> Dict[str, int]:
    """Compute metrics"""

    # create dicts
    gold_labels_dict = {f'L{idx + 1}': [] for idx in range(len(gold_labels[0]))}
    predictions_dict = {f'L{idx + 1}': [] for idx in range(len(predictions[0]))}

    for row in gold_labels:
        for col_idx, item in enumerate(row):
            gold_labels_dict[f'L{col_idx + 1}'].append(item)

    for row in predictions:
        for col_idx, item in enumerate(row):
            predictions_dict[f'L{col_idx + 1}'].append(item)

    scores_dict = {
        'acc': [],
        'f1': [],
        'precision': [],
        'recall': [],
    }
    for idx in range(len(gold_labels[0])):
        scores_dict['acc'].append(accuracy_score(gold_labels_dict[f'L{idx + 1}'], predictions_dict[f'L{idx + 1}']))
        scores_dict['f1'].append(
            f1_score(gold_labels_dict[f'L{idx + 1}'], predictions_dict[f'L{idx + 1}'], labels=all_classes,
                     average='weighted'))
        scores_dict['precision'].append(
            f1_score(gold_labels_dict[f'L{idx + 1}'], predictions_dict[f'L{idx + 1}'], labels=all_classes,
                     average='weighted'))
        scores_dict['recall'].append(
            f1_score(gold_labels_dict[f'L{idx + 1}'], predictions_dict[f'L{idx + 1}'], labels=all_classes,
                     average='weighted'))

    return {k: np.mean([v[desired_idx] for desired_idx in desired_label_ids]) for k, v in scores_dict.items()}
