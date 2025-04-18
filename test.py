from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

repo = "yuchenxie/arlowgpt-dummy-weights"

cfg = AutoConfig.from_pretrained(repo)

# ➊  build an *empty* shell on the meta device
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg)

# ➋  stream weights straight onto GPU 0 (or "auto" for multi‑GPU)
model = load_checkpoint_and_dispatch(
    model,
    repo,
    device_map={"": "cuda:0"},          # or "auto"
    no_split_module_classes=["ArlowFlashTransformerLayer"],  # optional
    dtype="bfloat16",
)
