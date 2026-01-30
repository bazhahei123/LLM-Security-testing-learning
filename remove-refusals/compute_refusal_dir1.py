import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from tqdm import tqdm

torch.inference_mode()

MODEL_ID = "/models/Qwen332B"
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
# MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             trust_remote_code=True,
                                             dtype=torch.float16,
                                             device_map="cuda",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                    bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 64
layer_idx = int(14)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful_64_webshell_rce_waf.txt", "r") as f:
    harmful = f.readlines()

with open("harmless_64_webshell_rce_waf.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
                                  add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
                                  add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)


def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device),
                          use_cache=False,
                          max_new_tokens=1,
                          return_dict_in_generate=True,
                          output_hidden_states=True)


harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

# --- debug: inspect hidden_states structure (print once) ---
o = harmful_outputs[0]
print("len(output.hidden_states) =", len(o.hidden_states))          # 通常 = 1 (因为 max_new_tokens=1)
print("len(output.hidden_states[0]) =", len(o.hidden_states[0]))    # 通常 = 64 或 65
print("shapes of first 3 layers:")
for i in range(3):
    print(i, o.hidden_states[0][i].shape)
print("last idx:", len(o.hidden_states[0]) - 1, o.hidden_states[0][-1].shape)
# ----------------------------------------------------------

bar.close()


harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

print(harmful_hidden)

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")

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

# hidden_states dtype 是最关键的对齐指标
toks = tokenizer("hello", return_tensors="pt")
toks = {k: v.to(model.device) for k, v in toks.items()}
with torch.no_grad():
    out = model(**toks, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[0]
    print("hidden_states[0] dtype:", hs.dtype, "shape:", tuple(hs.shape))
print("=============================")

