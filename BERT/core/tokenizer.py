import re
from typing import List, Dict, Optional, Tuple

class BertTokenizer:
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        do_lower_case: bool = True,
        max_len: int = 512,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
    ):
        self.do_lower_case = do_lower_case
        self.max_len = max_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.vocab = {}
        self.ids_to_tokens = {}

        if vocab_file:
            self._load_vocab(vocab_file)
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.sep_token,
            self.cls_token,
            self.mask_token,
        ]

        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ids_to_tokens[i] = token

    def _load_vocab(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = i
                self.ids_to_tokens[i] = token

    def tokenize(self, text: str) -> List[str]:
        if self.do_lower_case:
            text = text.lower()

        tokens = self._basic_tokenize(text)
        tokens = self._wordpiece_tokenize(tokens)

        return tokens

    def _basic_tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        new_tokens = []
        for token in tokens:
            sub_tokens = re.findall(r'\w+|[^\w\s]', token)
            new_tokens.extend(sub_tokens)

        return new_tokens

    def _wordpiece_tokenize(self, tokens: List[str]) -> List[str]:
        output_tokens = []
        for token in tokens:
            chars = list(token)
            if len(chars) > self.max_len:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None

                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Dict[str, List[int]]:
        if max_length is None:
            max_length = self.max_len

        tokens = self.tokenize(text)
        if text_pair:
            tokens_pair = self.tokenize(text_pair)
        else:
            tokens_pair = []

        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
            if tokens_pair:
                tokens += tokens_pair + [self.sep_token]

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        if padding:
            padding_length = max_length - len(input_ids)
            input_ids += [self.vocab[self.pad_token]] * padding_length
            attention_mask += [0] * padding_length

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
