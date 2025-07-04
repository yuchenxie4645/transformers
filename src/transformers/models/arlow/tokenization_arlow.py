import json
import os
from typing import Optional

import regex as re

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 / ByteLevel BPE uses a list of utf-8 bytes and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings. This function and the reversible bpe codes
    allow us to simulate 'byte-level' subwords in purely unicode space.
    """
    # All the printable characters from ASCII plus some more, as used in GPT-2:
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: list[str]) -> set:
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ArlowTokenizer(PreTrainedTokenizer):
    """
    ArlowTokenizer is a custom slow BPE tokenizer for the ArlowGPT model,
    implementing a GPT-2/ByteLevel-like approach in pure Python.

    Example:
        >>> from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
        >>> tokenizer = ArlowTokenizer.from_pretrained("path/to/tokenizer")
        >>> tokens = tokenizer("Hello, world!")
        >>> print(tokens)

    Attributes:
        vocab_file (str): Path to the vocabulary file (JSON).
        merges_file (str): Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`): The beginning of sequence token that was used during pretraining.
            Can be used as a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`): The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<|unk|>"`): The unknown token. A token that is not in the vocabulary
            cannot be converted to an ID and is set to be this token instead.
        pad_token (`str`, *optional*, defaults to `"<|pad|>"`): The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<|mask|>"`): The token used for masking values. This is the token used when training
            this model with masked language modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*): Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    fast_tokenizer_class = "ArlowTokenizerFast"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: str = "<|unk|>",
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        additional_special_tokens: Optional[list[str]] = None,
        **kwargs,
    ):
        # Store file paths for saving/conversion
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # Convert or extend any additional special tokens
        default_special_tokens = []
        if additional_special_tokens and isinstance(additional_special_tokens, list):
            default_special_tokens.extend(additional_special_tokens)

        # Load vocabulary
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Load merges (BPE ranks)
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")
            merges = [m for m in merges if m and not m.startswith("#")]
            self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(merges)}

        # Byte-level mapping (GPT-2 style)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Cache for subwords
        self.cache = {}

        # A regex matching GPT-2/ByteLevel style word boundaries
        # This will match:
        #   - English contractions ('s, 't, etc.)
        #   - Letters and digits
        #   - Symbols
        #   - Whitespace blocks
        # You can adjust as needed for more or less aggressive splitting
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=default_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.encoder)

    def bpe(self, token: str) -> str:
        """
        Given a 'word' in the ByteLevel space, perform BPE merges according to self.bpe_ranks.
        """
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        pairs = get_pairs(word)
        if not pairs:
            self.cache[token] = token
            return token

        while True:
            # Find the highest-ranked (lowest index) bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                j = -1
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        subword = " ".join(word)
        self.cache[token] = subword
        return subword

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize the text into ByteLevel subwords, then apply BPE merges.
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 1. Encode each character into a 'safe' unicode
            token_bytes = token.encode("utf-8")
            chars = "".join(self.byte_encoder[b] for b in token_bytes)
            # 2. Apply BPE merges
            for sub in self.bpe(chars).split(" "):
                bpe_tokens.append(sub)
        return bpe_tokens

    def _convert_token_to_id(self, token: str) -> int:
        # Return the ID if found, else the <unk> token ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Reconstructs the text by reversing the ByteLevel encoding. We map each subword
        back to original bytes, then decode to UTF-8.
        """
        text = "".join(tokens)
        # Convert each character in the text back to its original byte
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        return byte_array.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str, str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return (None, None)

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # Save the vocab.json
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        # Save the merges.txt
        with open(merges_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # Sort merges by their BPE rank
            merges_sorted = sorted(self.bpe_ranks.items(), key=lambda kv: kv[1])
            for i, ((first, second), rank) in enumerate(merges_sorted):
                if i > 0:
                    writer.write("\n")
                writer.write(f"{first} {second}")

        return vocab_file, merges_file


__all__ = ["ArlowTokenizer"]
