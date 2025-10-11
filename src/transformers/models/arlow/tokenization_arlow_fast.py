from typing import Optional

from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from tokenizers.pre_tokenizers import ByteLevel

from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
from transformers.tokenization_utils import AddedToken
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",  # For a full fast tokenizer JSON.
}


class ArlowTokenizerFast(PreTrainedTokenizerFast):
    """
    ArlowTokenizerFast is a custom fast tokenizer for the ArlowGPT model.
    It is backed by Hugging Face’s fast tokenizer library and supports subword tokenization,
    padding, truncation, and special token handling.

    Example:
        >>> from transformers.models.arlow.tokenization_arlow_fast import ArlowTokenizerFast
        >>> tokenizer = ArlowTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> tokens = tokenizer("Hello, world!")
        >>> print(tokens)
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = ArlowTokenizer
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<|unk|>",
        bos_token: Optional[str] = "<|startoftext|>",
        eos_token: str = "<|im_end|>",  # Turn boundary serves as EOS (Qwen-style)
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        additional_special_tokens: Optional[list] = None,
        add_bos_token: bool = False,  # Chat template controls special tokens
        add_eos_token: bool = False,  # Chat template controls special tokens
        **kwargs,
    ):
        # Convert str tokens to AddedToken objects with no normalization,
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
        mask_token = (
            AddedToken(mask_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # Initialize via the parent class. This will load tokenizer.json if provided
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # Expose BOS/EOS behavior for explicit control
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        # If there's NO tokenizer_file, we're building from vocab/merges. Ensure ByteLevel is set:
        if tokenizer_file is None:
            # Force ByteLevel pre-tokenizer + decoder for GPT-2 style byte handling
            # add_prefix_space=False follows modern causal LM standards (CodeGen, StarCoder, DeepSeek)
            if self._tokenizer.pre_tokenizer is None:
                self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            if self._tokenizer.decoder is None:
                self._tokenizer.decoder = ByteLevelDecoder()

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens.

        Args:
            token_ids_0 (`List[int]`): The first sequence to be encoded.
            token_ids_1 (`List[int]`, *optional*): The second sequence to be encoded (for sequence pairs).

        Returns:
            `List[int]`: The encoded sequence(s) with special tokens.
        """
        bos = [self.bos_token_id] if getattr(self, "add_bos_token", False) and self.bos_token_id is not None else []

        if token_ids_1 is None:
            return bos + token_ids_0

        # For pairs, add BOS to the first sequence only
        return bos + token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Save the vocabulary files (vocab.json, merges.txt) from the underlying tokenizers library.
        This is used if you want the 'slow' version of the tokenizer, but it works even if
        you're using a fast tokenizer. It extracts the BPE merges and vocab from the
        Rust tokenizer.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


__all__ = ["ArlowTokenizerFast"]