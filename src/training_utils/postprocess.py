import logging
from typing import List

from data_utils import LabelField

logger = logging.getLogger(__name__)


def postprocess_labels(labels: List[List[int]], label_fields: List[LabelField]) -> List[List[str]]:
    """Postprocess labels. Convert ints to corresponding strings."""
    out = []
    for li in labels:
        out.append([label_fields[idx].itos[li[idx]] for idx in range(len(label_fields))])
    return out
