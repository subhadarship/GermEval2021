import logging
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from .data import POSSIBLE_LABEL_COLUMN_NAMES
from .field import LabelField

logger = logging.getLogger(__name__)


class BertGermEvalDataset(Dataset):
    """Bert GermEval dataset"""

    def __init__(self, df: pd.DataFrame, label_col_names: List[str], bert_tokenizer: BertTokenizer,
                 label_fields: List[LabelField]):

        # sanity check
        assert len(set(POSSIBLE_LABEL_COLUMN_NAMES).union(set(label_col_names))) == 3

        self.label_col_names = label_col_names
        self.bert_tokenizer = bert_tokenizer
        self.label_fields = label_fields

        self.all_original_sentences = df['comment_text'].astype(str).tolist()
        self.labels = {
            f'L{idx + 1}': df[label_col_name].astype(str).tolist() for idx, label_col_name in
            enumerate(self.label_col_names)
        }

        self.all_sent_ids = []
        self.all_label_ids = []
        for sample_idx, sentence in enumerate(
                tqdm(self.all_original_sentences, desc='prepare bert data', unit=' samples')):
            ids = self.bert_tokenizer.encode(sentence)  # [CLS idx, ..., SEP idx]
            if len(ids) > self.bert_tokenizer.model_max_length:
                logger.warning(
                    f'trimming sentence {sample_idx} of length {len(ids)} to {self.bert_tokenizer.model_max_length} tokens '
                    f'(trimmed tokens include {self.bert_tokenizer.cls_token} and {self.bert_tokenizer.sep_token} tokens)'
                )
                ids = ids[:self.bert_tokenizer.model_max_length - 1] + [self.bert_tokenizer.sep_token_id]

            self.all_sent_ids.append(torch.LongTensor(ids))
            label_ids = {}
            for idx in range(len(self.label_col_names)):
                label_ids[f'L{idx + 1}'] = torch.LongTensor(
                    [self.label_fields[idx].stoi[self.labels[f'L{idx + 1}'][sample_idx]]]
                )
            self.all_label_ids.append(label_ids)

    def __getitem__(self, idx):
        return {
            'text': self.all_sent_ids[idx],
            'labels': self.all_label_ids[idx],
            'orig': self.all_original_sentences[idx],
            'orig_labels': ' '.join(
                [self.labels[f'L{label_idx + 1}'][idx] for label_idx in range(len(self.label_col_names))]),
        }

    def __len__(self):
        return len(self.all_sent_ids)
