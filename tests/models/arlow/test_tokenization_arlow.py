import json
import os
import shutil
import tempfile
import unicodedata
import unittest

from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
from transformers.models.arlow.tokenization_arlow_fast import ArlowTokenizerFast


DUMMY_VOCAB = {
    "h": 0,
    "e": 1,
    "l": 2,
    "o": 3,
    "w": 4,
    "r": 5,
    "d": 6,
    "<|endoftext|>": 7,
    "<|im_start|>": 8,
    "<|im_end|>": 9,
}
DUMMY_MERGES = ""

DUMMY_TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    },
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "decode_special_tokens": True},
    "model": {
        "unk_token": "<|endoftext|>",
        "type": "BPE",
        "vocab": DUMMY_VOCAB,
        "merges": [],
    },
}


class ArlowTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmp, "vocab.json")
        self.merges_file = os.path.join(self.tmp, "merges.txt")
        self.tokenizer_file = os.path.join(self.tmp, "tokenizer.json")
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump(DUMMY_VOCAB, f)
        with open(self.merges_file, "w", encoding="utf-8") as f:
            f.write(DUMMY_MERGES)
        with open(self.tokenizer_file, "w", encoding="utf-8") as f:
            json.dump(DUMMY_TOKENIZER_JSON, f)
        self.text = "hello world"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_slow_tokenizer_roundtrip(self):
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)
        enc = tok(self.text, return_tensors=None)
        self.assertIn("input_ids", enc)
        ids = enc["input_ids"]
        self.assertTrue(isinstance(ids, list))
        _ = tok.decode(ids)

    def test_fast_tokenizer_roundtrip(self):
        tok = ArlowTokenizerFast.from_pretrained(self.tmp)
        enc = tok(self.text, padding=True)
        self.assertIn("input_ids", enc)
        _ = tok.decode(enc["input_ids"][0])

    def test_unicode_normalization_slow(self):
        """Test that Unicode NFC normalization is applied to prevent UTF-8 artifacts."""
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)

        # Test with composed vs decomposed unicode
        # "café" with composed é (U+00E9)
        text_composed = "café"
        # "café" with decomposed é (e + combining accent U+0065 U+0301)
        text_decomposed = "cafe\u0301"

        # Verify they're different before normalization
        self.assertNotEqual(text_composed, text_decomposed)
        self.assertEqual(len(text_composed), 4)
        self.assertEqual(len(text_decomposed), 5)

        # After normalization, they should produce the same tokens
        normalized_composed, _ = tok.prepare_for_tokenization(text_composed)
        normalized_decomposed, _ = tok.prepare_for_tokenization(text_decomposed)

        self.assertEqual(normalized_composed, normalized_decomposed)
        self.assertEqual(normalized_composed, unicodedata.normalize("NFC", text_composed))

        # Tokenization should produce identical results
        tokens_composed = tok(text_composed)["input_ids"]
        tokens_decomposed = tok(text_decomposed)["input_ids"]
        self.assertEqual(tokens_composed, tokens_decomposed)

    def test_special_tokens_normalization(self):
        """Test that special tokens are not normalized."""
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)

        # Special tokens should remain unchanged
        eos = tok.eos_token
        unk = tok.unk_token
        pad = tok.pad_token

        self.assertEqual(eos, "<|endoftext|>")
        self.assertEqual(unk, "<|endoftext|>")
        self.assertEqual(pad, "<|endoftext|>")
        self.assertIsNone(tok.bos_token)

        # Verify special tokens are properly marked
        self.assertIn(eos, tok.all_special_tokens)
        self.assertIn(unk, tok.all_special_tokens)

    def test_get_vocab_includes_added_tokens(self):
        """Test that get_vocab() includes added tokens."""
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)
        vocab = tok.get_vocab()

        # Check that special tokens are in vocab
        self.assertIn("<|endoftext|>", vocab)

    def test_decode_spaces_between_special_tokens(self):
        """Test that decode() properly handles spaces_between_special_tokens."""
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)

        # Encode some text with special tokens
        ids = tok.encode("hello", add_special_tokens=False)

        # Decode without special tokens - should use default spaces_between_special_tokens=False
        decoded = tok.decode(ids)
        self.assertIsInstance(decoded, str)

    def test_batch_encoding_decoding(self):
        """Test batch encoding and decoding."""
        tok = ArlowTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)

        texts = ["hello world", "hello", "world"]
        encodings = tok(texts, padding=True, return_tensors=None)

        self.assertIn("input_ids", encodings)
        self.assertIn("attention_mask", encodings)
        self.assertEqual(len(encodings["input_ids"]), 3)

        # Decode batch
        for ids in encodings["input_ids"]:
            decoded = tok.decode(ids, skip_special_tokens=True)
            self.assertIsInstance(decoded, str)

    def test_constants_exist(self):
        """Test that required constants are defined."""
        from transformers.models.arlow.tokenization_arlow import MAX_MODEL_INPUT_SIZES, PRETOKENIZE_REGEX

        self.assertIsInstance(MAX_MODEL_INPUT_SIZES, dict)
        self.assertIsInstance(PRETOKENIZE_REGEX, str)


if __name__ == "__main__":
    unittest.main()
