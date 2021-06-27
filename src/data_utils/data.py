import logging
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .field import Field, LabelField

logger = logging.getLogger(__name__)

POSSIBLE_LABEL_COLUMN_NAMES = ['Sub1_Toxic', 'Sub2_Engaging', 'Sub3_FactClaiming']


class GermEvalDataset(Dataset):
    """GermEval dataset"""

    def __init__(self, df: pd.DataFrame, label_col_names: List[str], text_field: Field, label_fields: List[LabelField],
                 build_vocab: bool,
                 max_len: Union[int, None]):

        # sanity check
        assert len(set(POSSIBLE_LABEL_COLUMN_NAMES).union(set(label_col_names))) == 3

        self.label_col_names = label_col_names
        self.text_field = text_field
        self.label_fields = label_fields
        self.build_vocab = build_vocab
        self.max_len = max_len

        self.all_original_sentences = df['comment_text'].astype(str).tolist()
        self.labels = {
            f'L{idx + 1}': df[label_col_name].astype(str).tolist() for idx, label_col_name in
            enumerate(self.label_col_names)
        }

        if self.build_vocab:
            # build vocab
            self.text_field.build_vocab(self.all_original_sentences)

        self.all_preprocessed_sentences = []
        self.all_sent_ids = []
        self.all_label_ids = []
        for sample_idx, sentence in enumerate(tqdm(self.all_original_sentences, desc='prepare data', unit=' samples')):
            preprocessed = self.text_field.preprocess(sentence)
            self.all_preprocessed_sentences.append(preprocessed)
            ids = self.text_field.convert_tokens_to_ids(preprocessed)
            if self.max_len is not None and len(ids) > self.max_len:
                orig_len = len(ids)
                if self.text_field.sos_token is not None and self.text_field.sos_token is not None:
                    log_note = f'(trimmed tokens include {self.text_field.sos_token} and {self.text_field.eos_token} tokens)'
                    ids = ids[:self.max_len - 1] + [ids[-1]]
                elif self.text_field.sos_token is not None:
                    log_note = f'(trimmed tokens include {self.text_field.sos_token} token)'
                    ids = ids[:self.max_len]
                elif self.text_field.eos_token is not None:
                    log_note = f'(trimmed tokens include {self.text_field.eos_token} token)'
                    ids = ids[:self.max_len - 1] + [ids[-1]]
                else:
                    log_note = f''
                    ids = ids[:self.max_len]

                logger.warning(
                    f'trimming sentence {sample_idx} of length {orig_len} to {self.max_len} tokens {log_note}'
                )
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
            'prepro': self.all_preprocessed_sentences[idx],
            'orig_labels': ' '.join(
                [self.labels[f'L{label_idx + 1}'][idx] for label_idx in range(len(self.label_col_names))]),
        }

    def __len__(self):
        return len(self.all_sent_ids)
