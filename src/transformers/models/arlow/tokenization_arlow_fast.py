from typing import Optional, Tuple, Union

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from .tokenization_arlow import PRETOKENIZE_REGEX, ArlowTokenizer
from ...tokenization_utils_base import AddedToken
from ...tokenization_utils_tokenizers import PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",  # For a full fast tokenizer JSON.
}

MAX_MODEL_INPUT_SIZES = {"arlow": 131072}


# Inspired by transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast
class ArlowTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Arlow tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import ArlowTokenizerFast

    >>> tokenizer = ArlowTokenizerFast.from_pretrained("arlow-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = ArlowTokenizer
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[Union[str, dict[str, int]]] = None,
        merges: Optional[Union[str, list[str]]] = None,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<|endoftext|>",
        bos_token: Optional[str] = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        add_prefix_space: Optional[bool] = None,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else False
        self._vocab = vocab if vocab is not None else (vocab_file if vocab_file is not None else {"<|endoftext|>": 0})
        self._merges = merges if merges is not None else (merges_file if merges_file is not None else [])
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

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=self.add_prefix_space,
            **kwargs,
        )
        additional_specials = [
            "<image>",
            "<video>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]
        try:
            self.add_special_tokens({"additional_special_tokens": additional_specials})
        except Exception:
            pass

        self.image_token_id = self.convert_tokens_to_ids("<image>")
        self.video_token_id = self.convert_tokens_to_ids("<video>")
        self.vision_start_token_id = self.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.convert_tokens_to_ids("<|vision_end|>")
        self.init_kwargs["image_token_id"] = self.image_token_id
        self.init_kwargs["video_token_id"] = self.video_token_id
        self.init_kwargs["vision_start_token_id"] = self.vision_start_token_id
        self.init_kwargs["vision_end_token_id"] = self.vision_end_token_id

    # Copied from transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(str(file_path) for file_path in files)


__all__ = ["ArlowTokenizerFast"]
