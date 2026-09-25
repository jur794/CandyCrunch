import json
import re
from collections import Counter

import torch


# Splits an IUPAC-condensed glycan into structural units:
#   - a complete linkage including its parentheses, e.g. "(a2-3)"
#   - a branch bracket, "[" or "]"
#   - everything in between, i.e. a monosaccharide with its modifications, e.g. "Neu5Ac", "GlcNAc6S"
# 'Gal(b1-3)[Neu5Ac(a2-6)]GalNAc' -> ['Gal', '(b1-3)', '[', 'Neu5Ac', '(a2-6)', ']', 'GalNAc']
UNIT_PATTERN = re.compile(r'\([^()\[\]{}]*\)?|[\[\]{}]|\)|[^()\[\]{}]+')

END_OF_UNIT = '</w>'


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for IUPAC-condensed glycan sequences.

    Each glycan is first split into structural units (monosaccharides, linkages,
    branch brackets). BPE merges are learned and applied *within* a unit only, so
    a token can never span a monosaccharide-linkage boundary. As a result:

    - every token is a (piece of a) monosaccharide, a (piece of a) linkage, or a bracket
    - linkage parentheses always open and close inside the same unit
    - modifications can still be composed freely (e.g. Glc + 6S + 3Me),
      and single characters remain as a fallback, so there are no <unk> tokens
      for characters seen during training

    Branch brackets ``[`` and ``]`` bypass BPE entirely and are always single tokens.
    """

    def __init__(self, structural_symbols=None, max_seq_length=512):
        self.special_tokens = {
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'mask_token': '<mask>'
        }
        self.structural = set(structural_symbols if structural_symbols is not None else ['[', ']', '{', '}'])
        self.merges = []
        self.vocab = {}
        self.reverse_vocab = {}
        self.max_seq_length = max_seq_length
        self._cache = {}

    # ------------------------------------------------------------------ #
    # Pre-tokenization
    # ------------------------------------------------------------------ #
    def _pre_tokenize(self, text):
        """Split a glycan string into monosaccharide, linkage and bracket units."""
        return UNIT_PATTERN.findall(text.strip())

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_pair_frequencies(unit_counts):
        pairs = Counter()
        for unit, freq in unit_counts.items():
            symbols = unit.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_symbols(pair, unit_counts):
        bigram = " ".join(pair)
        replacement = "".join(pair)
        # Only match the pair as two *whole* symbols, never as part of a larger symbol
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        merged_counts = {}
        for unit, freq in unit_counts.items():
            new_unit = pattern.sub(replacement, unit)
            merged_counts[new_unit] = merged_counts.get(new_unit, 0) + freq
        return merged_counts

    def train(self, corpus, target_vocab_size):
        """Learn the subword vocabulary and BPE merge rules from a list of glycan strings."""
        # Every non-bracket unit becomes a space-separated character sequence
        # terminated by an end-of-unit marker: 'Neu5Ac' -> 'N e u 5 A c </w>'
        unit_counts = Counter()
        for text in corpus:
            for unit in self._pre_tokenize(text):
                if unit in self.structural:
                    continue
                unit_counts[" ".join(unit) + " " + END_OF_UNIT] += 1

        alphabet = set()
        for unit in unit_counts:
            alphabet.update(unit.split())

        # Structural symbols are guaranteed vocabulary tokens, even if the corpus lacks them
        current_vocab = list(self.special_tokens.values()) + sorted(alphabet | self.structural)
        self.merges = []

        while len(current_vocab) < target_vocab_size:
            pairs = self._get_pair_frequencies(unit_counts)
            if not pairs:
                break  # every unit is already a single token
            best_pair = max(pairs, key=pairs.get)
            unit_counts = self._merge_symbols(best_pair, unit_counts)
            self.merges.append(best_pair)
            current_vocab.append("".join(best_pair))

        self.vocab = {token: idx for idx, token in enumerate(current_vocab)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self._cache = {}

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def _bpe_encode_unit(self, unit):
        """Apply the learned merges, in order, to a single unit."""
        if unit in self._cache:
            return self._cache[unit]

        symbols = list(unit) + [END_OF_UNIT]
        for first, second in self.merges:
            if len(symbols) == 1:
                break
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        self._cache[unit] = symbols
        return symbols

    def tokenize(self, text):
        tokens = []
        for unit in self._pre_tokenize(text):
            if unit in self.structural:
                tokens.append(unit)
            else:
                tokens.extend(self._bpe_encode_unit(unit))
        return tokens

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        unk_id = self.vocab[self.special_tokens['unk_token']]
        bos_id = self.vocab[self.special_tokens['bos_token']]
        eos_id = self.vocab[self.special_tokens['eos_token']]
        pad_id = self.vocab[self.special_tokens['pad_token']]

        batch_token_ids = []
        batch_attention_masks = []

        for text in texts:
            tokens = self.tokenize(text)
            token_ids = [self.vocab.get(token, unk_id) for token in tokens]
            token_ids = [bos_id] + token_ids + [eos_id]
            attention_mask = [1] * len(token_ids)

            if len(token_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - len(token_ids)
                token_ids += [pad_id] * padding_length
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

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detokenize(tokens):
        # IUPAC-condensed has no spaces: strip end-of-unit markers and join directly
        return "".join(token.replace(END_OF_UNIT, "") for token in tokens)

    def decode(self, batch_token_ids, skip_special_tokens=False):
        if batch_token_ids.dim() == 1:
            batch_token_ids = batch_token_ids.unsqueeze(0)

        pad_id = self.vocab[self.special_tokens['pad_token']]
        special_ids = {self.vocab[val] for val in self.special_tokens.values()}

        decoded_texts = []
        for token_ids in batch_token_ids:
            if skip_special_tokens:
                kept_ids = [tid.item() for tid in token_ids if tid.item() not in special_ids]
            else:
                kept_ids = [tid.item() for tid in token_ids if tid.item() != pad_id]
            tokens = [self.reverse_vocab[tid] for tid in kept_ids]
            decoded_texts.append(self._detokenize(tokens))

        return decoded_texts if len(decoded_texts) > 1 else decoded_texts[0]

    # ------------------------------------------------------------------ #
    # Saving / loading
    # ------------------------------------------------------------------ #
    def save_vocabulary(self, path="vocab.json"):
        with open(path, 'w') as file:
            json.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "structural": sorted(self.structural),
                "max_seq_length": self.max_seq_length,
                "pre_tokenization": "units"
            }, file)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def load_vocabulary(cls, path="vocab.json"):
        with open(path, 'r') as file:
            saved = json.load(file)
        if saved.get("pre_tokenization") != "units":
            raise ValueError(f"{path} was trained with the old tokenizer (no unit pre-tokenization). "
                             "Retrain the vocabulary with this version.")
        tokenizer = cls(structural_symbols=saved["structural"], max_seq_length=saved["max_seq_length"])
        tokenizer.vocab = saved["vocab"]
        tokenizer.reverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(pair) for pair in saved["merges"]]
        return tokenizer