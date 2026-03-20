from __future__ import annotations

import ast
import json
import os
import unicodedata

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"arlow": 131072}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def _read_vocab_file(vocab_file: str) -> dict[str, int]:
    with open(vocab_file, encoding="utf-8") as vocab_handle:
        text = vocab_handle.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Keep supporting simple Python dict strings used by local tests.
        return ast.literal_eval(text)


def _resolve_vocab(
    vocab: str | dict[str, int] | None,
    vocab_file: str | None,
) -> dict[str, int]:
    if isinstance(vocab, dict):
        return vocab
    if isinstance(vocab, str) and os.path.isfile(vocab):
        return _read_vocab_file(vocab)
    if vocab_file is not None and os.path.isfile(vocab_file):
        return _read_vocab_file(vocab_file)
    return {"<|endoftext|>": 0}


def _read_merges_file(merges_file: str) -> list[tuple[str, str]]:
    bpe_merges = []
    with open(merges_file, encoding="utf-8") as merges_handle:
        for i, line in enumerate(merges_handle):
            line = line.strip()
            if (i == 0 and line.startswith("#version:")) or not line:
                continue
            first, second = line.split()
            bpe_merges.append((first, second))
    return bpe_merges


def _resolve_merges(
    merges: str | list[str] | list[tuple[str, str]] | None,
    merges_file: str | None,
) -> list[tuple[str, str]]:
    if isinstance(merges, str) and os.path.isfile(merges):
        return _read_merges_file(merges)
    if isinstance(merges, list):
        if not merges:
            return []
        if isinstance(merges[0], str):
            return [tuple(merge.split()) for merge in merges]
        return [tuple(merge) for merge in merges]
    if merges_file is not None and os.path.isfile(merges_file):
        return _read_merges_file(merges_file)
    return []


def _filter_invalid_merges(vocab: dict[str, int], merges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Be permissive with lightweight or partially-specified test vocabularies.
    The Rust BPE backend rejects merges whose source or merged tokens are absent
    from the vocab, while the previous Python implementation simply stored them.
    """
    filtered_merges = []
    for first, second in merges:
        if first in vocab and second in vocab and first + second in vocab:
            filtered_merges.append((first, second))
    return filtered_merges


# Inspired by transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
class ArlowTokenizer(TokenizersBackend):
    """
    Construct an Arlow tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import ArlowTokenizer

    >>> tokenizer = ArlowTokenizer.from_pretrained("arlow-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    This is expected.

    You should not use GPT2Tokenizer instead, because of the different pretokenization rules.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary or path to the vocabulary file.
        merges (`str` or `list[str]` or `list[tuple[str, str]]`, *optional*):
            Custom merges list or path to the merges file.
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to a serialized `tokenizer.json` file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Kept for backward compatibility with the previous Arlow tokenizer API.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | list[tuple[str, str]] | None = None,
        vocab_file: str | None = None,
        merges_file: str | None = None,
        tokenizer_file: str | None = None,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str | None = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        add_prefix_space: bool | None = None,
        clean_up_tokenization_spaces: bool = False,
        split_special_tokens: bool = False,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else False
        self.errors = errors
        self._tokenizer = None

        if any(value is not None for value in (vocab, merges, vocab_file, merges_file)) or (
            tokenizer_file is None or not os.path.isfile(tokenizer_file)
        ):
            self._vocab = _resolve_vocab(vocab, vocab_file)
            self._merges = _resolve_merges(merges, merges_file)
            self._merges = _filter_invalid_merges(self._vocab, self._merges)
            self._tokenizer = Tokenizer(
                BPE(
                    vocab=self._vocab,
                    merges=self._merges,
                    dropout=None,
                    unk_token=str(unk_token) if unk_token is not None else None,
                    continuing_subword_prefix="",
                    end_of_word_suffix="",
                    fuse_unk=False,
                    byte_fallback=False,
                )
            )
            self._tokenizer.decoder = decoders.ByteLevel()
            self._tokenizer.normalizer = normalizers.NFC()
            self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(
                        Regex(PRETOKENIZE_REGEX),
                        behavior="isolated",
                        invert=False,
                    ),
                    pre_tokenizers.ByteLevel(
                        add_prefix_space=self.add_prefix_space,
                        use_regex=False,
                    ),
                ]
            )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=self.add_prefix_space,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        self.image_token = "<image>"
        self.video_token = "<video>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(self.image_token, lstrip=False, rstrip=False, normalized=False, special=True),
                    AddedToken(self.video_token, lstrip=False, rstrip=False, normalized=False, special=True),
                    AddedToken(self.vision_start_token, lstrip=False, rstrip=False, normalized=False, special=True),
                    AddedToken(self.vision_end_token, lstrip=False, rstrip=False, normalized=False, special=True),
                ]
            }
        )

        self.image_token_id = self.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.convert_tokens_to_ids(self.video_token)
        self.vision_start_token_id = self.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.convert_tokens_to_ids(self.vision_end_token)
        self.init_kwargs["image_token_id"] = self.image_token_id
        self.init_kwargs["video_token_id"] = self.video_token_id
        self.init_kwargs["vision_start_token_id"] = self.vision_start_token_id
        self.init_kwargs["vision_end_token_id"] = self.vision_end_token_id

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()

        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(str(file_path) for file_path in files)

    def prepare_for_tokenization(self, text, **kwargs):
        """
        Performs NFC normalization before tokenization so composed and decomposed Unicode inputs
        are treated identically.
        """
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)


__all__ = ["ArlowTokenizer"]
