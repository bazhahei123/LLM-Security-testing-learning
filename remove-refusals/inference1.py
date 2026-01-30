import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from inspect import signature

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "/models/Qwen332B"
REFUSAL_PT = "/workspace/remove-refusals-with-transformers-master/_models_Qwen332B_refusal_dir.pt"

# Debug controls (avoid log spam)
DEBUG_PROJ_PRINTS = 5          # how many times to print projection before/after
DEBUG_LAYER_PRINTS = 5         # how many times to print "ablation layer called"
DEBUG_INSERT_PRINT = True      # print total layers after insert

# Generation controls
MAX_NEW_TOKENS = 2048

# ----------------------------
# Load model + tokenizer
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,   # transformers uses torch_dtype
    device_map="cuda",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# ----------------------------
# Load refusal direction vector
# ----------------------------
refusal_dir = torch.load(REFUSAL_PT, map_location="cpu")
refusal_dir = refusal_dir.squeeze()  # [1, 5120] -> [5120]
refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-12)

# Global debug counters
_DEBUG_proj_cnt = 0
_DEBUG_layer_cnt = 0


def direction_ablation_hook(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove the component of activation that lies along the 'direction' vector.
    Prints projection before/after for first DEBUG_PROJ_PRINTS calls.
    """
    global _DEBUG_proj_cnt

    d = direction.to(device=activation.device, dtype=activation.dtype)
    d = d / (d.norm() + 1e-12)

    # Projection
    coeff = (activation * d).sum(dim=-1, keepdim=True)  # [..., 1]
    proj = coeff * d                                   # [..., d_act]

    # Standard ablation (strength = 1.0)
    STRENGTH = 1.8
    if not hasattr(direction_ablation_hook, "_printed_strength"):
        print("STRENGTH =", STRENGTH)
        direction_ablation_hook._printed_strength = True
    out = activation - STRENGTH * proj

    # ---- DEBUG: projection before/after (first few calls only) ----
    if _DEBUG_proj_cnt < DEBUG_PROJ_PRINTS:
        before = (activation * d).sum(dim=-1).mean().item()
        after = (out * d).sum(dim=-1).mean().item()
        print(f"[proj before/after] {before:.6f} -> {after:.6f}")
        _DEBUG_proj_cnt += 1

    return out


# Determine whether model layers return Tensor or tuples (varies across implementations)
sig = signature(model.model.layers[0].forward)
simple = sig.return_annotation == torch.Tensor


class AblationDecoderLayer(nn.Module):
    """
    A lightweight layer that ablates (removes) the refusal direction from hidden_states.
    Inserted before each original decoder layer.
    """
    def __init__(self):
        super().__init__()
        self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        global _DEBUG_layer_cnt

        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir)

        # ---- DEBUG: confirm layer call (first few only) ----
        if _DEBUG_layer_cnt < DEBUG_LAYER_PRINTS:
            print(
                f"[ablation layer called] hidden_states={tuple(hidden_states.shape)} "
                f"device={hidden_states.device} dtype={hidden_states.dtype}"
            )
            _DEBUG_layer_cnt += 1

        if simple:
            return ablated

        outputs = (ablated,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


# ----------------------------
# Insert ablation layers
# ----------------------------
for idx in reversed(range(len(model.model.layers))):
    model.model.layers.insert(idx, AblationDecoderLayer())

if DEBUG_INSERT_PRINT:
    print("[debug] layers after insert:", len(model.model.layers))  # expect 128 if original is 64

# Keep config consistent with modified structure
if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers *= 2

# ----------------------------
# Precision / quant check (optional but useful)
# ----------------------------
print("=== precision/quant check ===")
print("torch:", torch.__version__)
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers: unknown", e)

try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except Exception as e:
    print("bitsandbytes: not found", e)

p = next(model.parameters())
print("param dtype:", p.dtype)
print("param class:", p.__class__)
print("device:", p.device)

if hasattr(model, "config"):
    print("config torch_dtype:", getattr(model.config, "torch_dtype", None))
    print("config model_type:", getattr(model.config, "model_type", None))

# Hidden states dtype / shape sanity check
toks0 = tokenizer("hello", return_tensors="pt")
toks0 = {k: v.to(model.device) for k, v in toks0.items()}
with torch.inference_mode():
    out0 = model(**toks0, output_hidden_states=True, use_cache=False)
    hs0 = out0.hidden_states[0]
    print("hidden_states[0] dtype:", hs0.dtype, "shape:", tuple(hs0.shape))
print("=============================")

# ----------------------------
# Chat loop (single-turn, no history)
# ----------------------------
streamer = TextStreamer(tokenizer)
print(f"Chat with {MODEL_ID} (single-turn, no history):")

while True:
    prompt = input().strip()
    if not prompt:
        continue

    conversation = [{"role": "user", "content": prompt}]

    toks = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Provide attention_mask to avoid warning / unstable behavior
    attention_mask = torch.ones_like(toks, dtype=torch.long)

    with torch.inference_mode():
        gen = model.generate(
            toks.to(model.device),
            attention_mask=attention_mask.to(model.device),
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    # ---- DEBUG: did we stop due to max_new_tokens or EOS? ----
    new_tokens = gen.shape[-1] - toks.shape[-1]
    eos = model.generation_config.eos_token_id
    last_id = gen[0, -1].item()

    print("\n[debug] new_tokens:", new_tokens, "max_new_tokens:", MAX_NEW_TOKENS)
    print("[debug] eos_token_id:", eos, "last_token_id:", last_id)

    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    # Not storing history (single-turn), but keeping structure in case you want it later

