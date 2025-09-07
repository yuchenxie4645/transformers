import json
import os
import shutil
import tempfile
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
    "<|unk|>": 7,
    "<|pad|>": 8,
    "<|startoftext|>": 9,
    "<|endoftext|>": 10,
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
        "unk_token": "<|unk|>",
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


if __name__ == "__main__":
    unittest.main()
