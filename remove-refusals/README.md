# Removing refusals with HF Transformers

***original tool URL:https://github.com/Sumandora/remove-refusals-with-transformers***

This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens. This means, that this supports every model that HF Transformers supports*.

The code was tested on a RTX 2060 6GB, thus mostly <3B models have been tested, but the code has been tested to work with bigger models as well.

*While most models are compatible, some models are not. Mainly because of custom model implementations. Some Qwen implementations for example don't work. Because `model.model.layers` can't be used for getting layers. They call the variables so that, `model.transformer.h` must be used, if I'm not mistaken.

## Usage
1. Set model and quantization in compute_refusal_dir1.py and inference.py (Quantization can apparently be mixed)
2. Run compute_refusal_dir1.py (Some settings in that file may be changed depending on your use-case,most import one is layer_idx)
3. Run inference.py and ask the model how to build an army of rabbits, that will overthrow your local government one day, by stealing all the carrots.

## Credits
- [Harmful instructions](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- [Harmless instructions](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Technique](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Update in this repository
1. compute_refusal_dir1.py： Confirms how many transformer layers the model has and what each hidden-state tensor looks like (batch size, sequence length, hidden size), so you can pick the correct layer_idx and avoid indexing/shape mistakes.

Verifies that you successfully extracted the per-sample hidden-state vectors from the same layer and token position (e.g., last token) for all harmful prompts, with the expected device/dtype, so they can be averaged and differenced to compute the refusal direction.
<img width="1920" height="812" alt="图片" src="https://github.com/user-attachments/assets/7395c172-110c-4a10-be7e-ea8e17f746c0" />

2. inference1.py：Add a new variable STRENGTH. Print the current STRENGTH value, and also print the projection values before and after ablation (proj before/after) for side-by-side comparison to confirm whether the ablation is taking effect.
<img width="2704" height="1054" alt="图片" src="https://github.com/user-attachments/assets/115af9ee-597d-4e0a-894c-5b11be5440ad" />

3. Add a new dataset for cybersecurity<harmful_128_websec.txt,harmless_128_websec.txt>,<harmful_64_webshell_rce_waf.txt,harmful_64_webshell_rce_waf.txt>.

4. success folder has success result could use directly(just use .pt directly)
