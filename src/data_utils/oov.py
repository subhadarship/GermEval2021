import logging
from typing import Union, Dict

from tqdm import tqdm

from .bert_data import BertGermEvalDataset
from .data import GermEvalDataset

logger = logging.getLogger(__name__)


def compute_num_unk(dataset: Union[GermEvalDataset, BertGermEvalDataset], unk_idx: int) -> Dict[str, int]:
    """Compute the number of running unk tokens and total tokens"""
    count_unk, count_total = 0, 0
    for sample in tqdm(dataset, desc='compute %unk', unit=' samples'):
        count_unk += sum(1 for token_idx in sample["text"] if token_idx == unk_idx)
        count_total += len(sample["text"])
    return {
        'running_unk_tokens': count_unk,
        'running_total_tokens': count_total,
    }
