import json
import torch
import re
from torch.utils.data import Dataset

# Token IDs
GLYCAN_VOCAB_DICT = {
    # Structural Symbols
    "(": 7, ")": 8, "[": 9, "]": 10,
    "{": 11, "}": 12, "/": 13, "?": 14,

    # Monosaccharide & Linkages
    "GalOMe": 15, "ManOS": 16, "Gal": 17, "Glc": 18, "HexNAc": 19,
    "GalNAcOPCho": 20, "Fuc": 21, "GalNAcOMe": 22, "GlcOS": 23,
    "IdoA2S": 24, "GlcNS6S": 25, ".1-3": 26, ".1-4": 27, "IdoA": 28,
    "HexA": 29, "GalNAc": 30, "GlcNAc6S": 31, "GalNAc4S": 32,
    "Gal6S": 33, "Ins": 34, "a2-.": 35, ".1-6": 36, "GalN": 37,
    "GlcOP": 38, "b1-3": 39, "GalNAc6S": 40, "GalNAcOS": 41,
    "Man": 42, "Glc-ol": 43, "Rha": 44, "ManOMe": 45, "GlcNAc": 46,
    "FucOS": 47, "1-.": 48, "a1-.": 49, "Ara": 50, "GlcN": 51,
    "GlcN6S": 52, "Xyl": 53, "Man6P": 54, "GlcNAcOS": 55,
    "HexNAcOS": 56, "b1-.": 57, "a1-4": 58, "GalOS": 59, "GlcA": 60,
    "Neu5Ac": 61, "HexA2S": 62, "Gal4S": 63, "GlcNS3S6S": 64,
    "a1-6": 65, "a2-8": 66, "a2-6": 67, "a1-3": 68, "Neu5Gc": 69,
    "a2-3": 70, "Rha3S": 71, "Hex": 72, "GlcNS3S": 73, "b1-4": 74,
    "b1-6": 75, ".1-.": 76, "b1-2": 77, "Gal3S": 78, "Neu5Ac8S": 79,
    "Kdn": 80, "a1-2": 81, "GlcNS": 82
}

class GlycoBartTokenizer:
    def __init__(self, vocab_dict=None, max_seq_length=512):
        if vocab_dict is None:
            vocab_dict = GLYCAN_VOCAB_DICT

        self.special_tokens = {
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '<sep>',
            'cls_token': '<cls>',
            'unk_token': '<unk>',
            'mask_token': '<mask>'
        }

        self.vocab = {}
        for idx, token in enumerate(self.special_tokens.values()):
            self.vocab[token] = idx

        for token, token_id in vocab_dict.items():
            if token not in self.vocab:
                self.vocab[token] = token_id

        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        pattern = r"([A-Za-z0-9]+|[\?\.]?\d+-[0-9/]+|[\(\)\[\]\{\}\?/])"
        tokens = re.findall(pattern, text)
        return tokens

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_token_ids = []
        batch_attention_masks = []

        for text in texts:
            tokens = self.tokenize(text)
            token_ids = [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']]) for token in tokens]

            token_ids = [self.vocab[self.special_tokens['bos_token']]] + token_ids + [self.vocab[self.special_tokens['eos_token']]]
            attention_mask = [1] * len(token_ids)

            if len(token_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - len(token_ids)
                token_ids += [self.vocab[self.special_tokens['pad_token']]] * padding_length
                attention_mask += [0] * padding_length
            else:
                token_ids = token_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]

            batch_token_ids.append(torch.tensor(token_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))

        return {
            "token_ids": torch.stack(batch_token_ids),
            "attention_mask": torch.stack(batch_attention_masks)
        }

    def decode(self, batch_token_ids, skip_special_tokens=True):
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.tolist()

        decoded_texts = []
        special_ids = {self.vocab[v] for v in self.special_tokens.values()} if skip_special_tokens else set()

        for seq in batch_token_ids:
            tokens = []
            for token_id in seq:
                if skip_special_tokens and token_id in special_ids:
                    if token_id == self.vocab[self.special_tokens['eos_token']]:
                        break
                    continue
                tokens.append(self.reverse_vocab.get(token_id, self.special_tokens['unk_token']))
            decoded_texts.append(" ".join(tokens))

        return decoded_texts