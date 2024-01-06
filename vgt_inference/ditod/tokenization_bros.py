# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""


import collections

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

BROS_BASE_UNCASED = "naver-clova-ocr/bros-base-uncased"
BROS_LARGE_UNCASED = "naver-clova-ocr/bros-large-uncased"
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        BROS_BASE_UNCASED: "https://huggingface.co/naver-clova-ocr/bros-base-uncased/resolve/main/vocab.txt",
        BROS_LARGE_UNCASED: "https://huggingface.co/naver-clova-ocr/bros-large-uncased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    BROS_BASE_UNCASED: 512,
    BROS_LARGE_UNCASED: 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    BROS_BASE_UNCASED: {"do_lower_case": True},
    BROS_LARGE_UNCASED: {"do_lower_case": True},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class BrosTokenizer(BertTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def tokenize_and_split_bboxs(self, tokens, bboxs):
        split_tokens = []
        split_bboxs = []
        token_ids = []
        for token, bbox in zip(tokens, bboxs):
            current_sub_tokens = []
            bboxs_start_x = bbox[0]
            bbox_length = bbox[2]
            tokens_split_on_whitespace = self.whitespace_tokenize(token)
            for split_token in tokens_split_on_whitespace:
                for sub_token in super().tokenize(split_token):
                    current_sub_tokens.append(sub_token)
                total_length_all_sub_tokens = sum(
                    [len(sub_token) for sub_token in current_sub_tokens]
                )
                for current_token in current_sub_tokens:
                    length_current_token = len(current_token)
                    current_bbox_width = bbox_length * (
                        length_current_token / total_length_all_sub_tokens
                    )
                    split_bboxs.append(
                        [bboxs_start_x, bbox[1], current_bbox_width, bbox[3]]
                    )
                    bboxs_start_x += current_bbox_width
                    split_tokens.append(current_token)
                    token_ids.append(super()._convert_token_to_id(current_token))
        final_outputs = {
            "input_ids": token_ids,
            "bbox_subword_list": split_bboxs,
            "texts": tokens,
            "bbox_texts_list": bboxs,
        }
        return final_outputs
