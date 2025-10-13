import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional

import regex as re

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

MAX_MODEL_INPUT_SIZES = {"arlow": 131072}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 / ByteLevel BPE uses a list of utf-8 bytes and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings. This function and the reversible bpe codes
    allow us to simulate 'byte-level' subwords in purely unicode space.
    """
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
    Construct an Arlow tokenizer. Based on byte-level Byte-Pair-Encoding.

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

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.tokenize("<|endoftext|>") =
            ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<|endoftext|>")` will be give `['<',
            '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only supported for `slow` tokenizers for the moment.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    fast_tokenizer_class = "ArlowTokenizerFast"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: Optional[str] = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        clean_up_tokenization_spaces: bool = False,
        split_special_tokens: bool = False,
        **kwargs,
    ):
        # Arlow vocab does not contain control tokens; added tokens need to be special
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # Load vocabulary
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.errors = errors  # how to handle errors in decoding

        # Load merges (BPE ranks)
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for i, line in enumerate(merges_handle):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # Byte-level mapping (GPT-2 style)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(PRETOKENIZE_REGEX)

        if kwargs.get("add_prefix_space", False):
            logger.warning_once(
                f"{self.__class__.__name__} does not support `add_prefix_space`, setting it to True has no effect."
            )

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.encoder, **self.added_tokens_encoder)

    @lru_cache(maxsize=4096)
    def bpe(self, token: str) -> str:
        """
        Given a 'word' in the ByteLevel space, perform BPE merges according to self.bpe_ranks.
        """
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
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
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = " ".join(word)
        return word

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
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces.
            spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add spaces between special tokens. Defaults to False to match modern tokenizers.
            **kwargs (additional keyword arguments):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded string.
        """
        # `spaces_between_special_tokens` defaults to True for _decode in slow tokenizers
        # and cannot be configured elsewhere, but it should default to False for ArlowTokenizer
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. An Arlow sequence has the following format:

        - single sequence: `X`
        - pair of sequences: `X Y`

        Args:
            token_ids_0 (`List[int]`): The first sequence to be encoded.
            token_ids_1 (`List[int]`, *optional*): The second sequence to be encoded (for sequence pairs).

        Returns:
            `List[int]`: The encoded sequence(s) with special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

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

    def prepare_for_tokenization(self, text, **kwargs):
        """
        Performs pre-tokenization normalization. This is critical to avoid UTF-8 byte artifacts
        and unicode inconsistencies during tokenizer training.

        Unicode NFC (Canonical Decomposition, followed by Canonical Composition) ensures that
        characters are represented in their composed form, preventing issues where:
        - "é" (single codepoint U+00E9) vs "é" (e + combining accent U+0065 U+0301)

        Without this normalization, these would produce different byte sequences and cause
        BPE to learn inconsistent merges, leading to tokenization artifacts.
        """
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)


__all__ = ["ArlowTokenizer"]
