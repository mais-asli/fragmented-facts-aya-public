"""Revision-pinned Aya loading and transparent, cache-free research inference."""
import importlib.metadata
import platform
import time
from pathlib import Path

from .hooks import capture_context, logit_lens, patch_context
from .io import read_json
from .scoring import continuation_ids, sequence_logprobs


class ModelRunner:
    def __init__(self, model, tokenizer, identity, max_new_tokens=24):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.identity = identity
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device
        self.context_limit = getattr(model.config, "max_position_embeddings", 8192)

    def inputs(self, ids):
        import torch
        if len(ids) > self.context_limit:
            raise ValueError("Input exceeds the model context limit")
        tensor = torch.tensor([ids], device=self.device, dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor), "use_cache": False}

    def logits(self, ids, patches=()):
        import torch
        with torch.inference_mode(), patch_context(self.model, patches):
            return self.model(**self.inputs(ids)).logits

    def score(self, prompt, answer, patches=()):
        ids = continuation_ids(self.tokenizer, prompt, answer)
        logits = self.logits(prompt.input_ids + ids, patches)
        score = sequence_logprobs(logits, len(prompt.input_ids), ids)
        del logits
        return {"answer": answer, **score}

    def generate(self, prompt, patches=()):
        """Greedy loop records raw token probabilities and correct truncation status.

        Recompute the prefix at each step; this is intentionally slower than KV
        caching, but gives identical patch semantics for every generation step.
        """
        import torch
        ids, generated, logprobs = list(prompt.input_ids), [], []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.discard(None)
        stopped = False
        started = time.perf_counter()
        for _ in range(self.max_new_tokens):
            logits = self.logits(ids, patches)[0, -1, :].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite generation logits")
            chosen = int(logits.argmax())
            lp = float(logits.log_softmax(-1)[chosen])
            generated.append(chosen)
            logprobs.append(lp)
            ids.append(chosen)
            if chosen in eos:
                stopped = True
                break
        answer_lp = [lp for token, lp in zip(generated, logprobs) if token not in eos]
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "generated_ids": generated, "generated_token_logprobs": logprobs,
            "answer_mean_logprob": sum(answer_lp) / len(answer_lp) if answer_lp else None,
            "answer_token_count": len(answer_lp), "truncated": not stopped,
            "seconds": time.perf_counter() - started}

    def capture(self, prompt, requests):
        with capture_context(self.model, requests) as cache:
            logits = self.logits(prompt.input_ids)
            del logits
        if set(cache) != set(requests):
            raise ValueError("Not all requested hooks executed")
        return cache

    def readout(self, prompt, layer, site, gold, distractor):
        import torch
        gold_ids = continuation_ids(self.tokenizer, prompt, gold)
        bad_ids = continuation_ids(self.tokenizer, prompt, distractor)
        if gold_ids[0] == bad_ids[0]:
            return {"eligible": False, "reason": "first_token_collision"}
        key = (layer, "residual", prompt.position(site))
        cache = self.capture(prompt, [key])
        with torch.inference_mode():
            z = logit_lens(self.model, cache[key])[0]
        return {"eligible": True, "gold_first_id": gold_ids[0], "distractor_first_id": bad_ids[0],
                "first_token_margin": float(z[gold_ids[0]] - z[bad_ids[0]])}

    def runtime(self):
        import torch
        packages = {}
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {"python": platform.python_version(), "packages": packages,
            "device": str(self.device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if self.device.type == "cuda" else None}


def load_runner(snapshot_manifest, config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    manifest = read_json(snapshot_manifest)
    if manifest["model_id"] != "CohereLabs/aya-23-8B":
        raise ValueError("The core pipeline requires Aya-23-8B; a second model needs a separate adapter/protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("Aya research runs require an allocated CUDA GPU. Use integration tests for CPU verification")
    if not Path(manifest["snapshot_path"]).is_dir():
        raise FileNotFoundError("Model snapshot missing on this machine; download/copy it and update the path")
    precision = config["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    compute_precision = config.get("nf4_compute_dtype", "fp16") if precision == "nf4" else precision
    if compute_precision not in ("fp16", "bf16"):
        raise ValueError("The compute dtype must be explicitly fp16 or bf16")
    if compute_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen compute dtype is BF16 but this GPU does not support it")
    dtype = torch.bfloat16 if compute_precision == "bf16" else torch.float16
    options = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                   device_map={"": 0}, attn_implementation=config.get("attention", "eager"))
    if precision == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    elif precision not in ("fp16", "bf16"):
        raise ValueError("Unknown precision")
    tok = AutoTokenizer.from_pretrained(manifest["snapshot_path"], use_fast=True, local_files_only=True, trust_remote_code=False)
    if not tok.is_fast:
        raise ValueError("Fast tokenizer required for subject offsets")
    model = AutoModelForCausalLM.from_pretrained(manifest["snapshot_path"], **options)
    if model.config.model_type != "cohere":
        raise ValueError("Unexpected architecture")
    identity = {"model_id": manifest["model_id"], "revision": manifest["revision"], "precision": precision,
                "compute_dtype": compute_precision, "attention": config.get("attention", "eager"), "data_kind": "research"}
    return ModelRunner(model, tok, identity, config.get("max_new_tokens", 24))
