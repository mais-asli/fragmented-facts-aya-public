"""Initial CPU NF4 inference using an authorized local Aya-23-8B snapshot.

This is a feasibility attempt, not E0 validation for the GPU research protocol.
Outputs and latency are saved after each example. No synthetic model fallback.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-manifest', required=True)
    parser.add_argument('--output', default='results/aya-cpu-smoke.json')
    parser.add_argument('--max-new-tokens', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--compute-dtype', choices=['fp16', 'bf16'], default='bf16')
    parser.add_argument('--check-hooks', action='store_true', help='Also check actual Aya activation hooks')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64 or args.threads < 1:
        parser.error('Use 1-64 output tokens and a positive thread count')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  model_id='CohereLabs/aya-23-8B', device='cpu', precision='nf4',
                  compute_dtype=args.compute_dtype, status='starting',
                  scope='Actual Aya CPU feasibility only; not GPU E0 or research results.',
                  weights_loaded=False, examples=[], python=platform.python_version(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        report['peak_process_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(target)

    save()
    try:
        import torch
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report['packages'] = {name: importlib.metadata.version(name) for name in
                              ['torch', 'bitsandbytes', 'transformers', 'accelerate', 'huggingface-hub']}
        report['threads'] = args.threads
        report['cpu_count'] = os.cpu_count()
        if Path('/proc/meminfo').exists():
            report['memory_before_load'] = {line.split(':')[0]: line.split(':')[1].strip()
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.split(':')[0] in ['MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree']}
        manifest = json.loads(Path(args.snapshot_manifest).read_text(encoding='utf-8'))
        if manifest['model_id'] != report['model_id']:
            raise ValueError('This entry point runs only CohereLabs/aya-23-8B')
        if len(manifest['revision']) != 40 or any(c not in '0123456789abcdef' for c in manifest['revision']):
            raise ValueError('An immutable model commit is required')
        snapshot = Path(manifest['snapshot_path'])
        config = json.loads((snapshot / 'config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'cohere' or config.get('hidden_size', 0) < 2048:
            raise ValueError('Snapshot does not describe the expected Aya-23 architecture')
        report['model_source'] = manifest
        report['status'] = 'loading'
        save()
        dtype = torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, device_map={'': 'cpu'}, torch_dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation='eager',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype),
        ).eval()
        report['load_seconds'] = time.perf_counter() - started
        report['nf4_linear_modules'] = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
        if report['nf4_linear_modules'] == 0:
            raise AssertionError('No 4-bit layers were loaded')
        report.update(weights_loaded=True, status='generating',
                      parameter_storage_gib=model.get_memory_footprint() / 2**30)
        save()
        if args.check_hooks:
            from fragmented_facts.hooks import capture_context, patch_context, Patch, logit_lens
            from fragmented_facts.io import code_hash
            report['research_code_hash'] = code_hash()
            report['status'] = 'checking_hooks'
            report['hook_checks'] = []
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content':
                'What is the capital of France? Answer with only the city name.'}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
            position = inputs['input_ids'].shape[-1] - 1
            middle, last = len(model.model.layers) // 2, len(model.model.layers) - 1
            requests = [(middle, component, position) for component in ['residual', 'mlp', 'attention']]
            requests.append((last, 'residual', position))
            with torch.inference_mode(), capture_context(model, requests) as captured:
                baseline = model(**inputs, use_cache=False).logits[:, -1, :].float()
            for layer, component, fixed_position in requests[:3]:
                patch = Patch(layer, component, fixed_position, captured[(layer, component, fixed_position)])
                with torch.inference_mode(), patch_context(model, [patch]):
                    observed = model(**inputs, use_cache=False).logits[:, -1, :].float()
                error = (observed - baseline).abs().max().item()
                passed = bool(torch.isfinite(observed).all()) and error <= 0.005
                report['hook_checks'].append(dict(name='identity_' + component, passed=passed,
                    layer=layer, position=fixed_position, max_logit_error=error))
                save()
                if not passed:
                    raise AssertionError('Actual Aya identity intervention failed')
            with torch.inference_mode():
                readout = logit_lens(model, captured[(last, 'residual', position)])
            error = (readout - baseline).abs().max().item()
            report['hook_checks'].append(dict(name='final_norm_logit_scale', passed=error <= 0.005,
                                               max_logit_error=error))
            save()
            if error > 0.005:
                raise AssertionError('Actual Aya final readout does not match the logits')
            zero = torch.zeros_like(captured[(last, 'residual', position)])
            with torch.inference_mode(), patch_context(model, [Patch(last, 'residual', position, zero)]):
                changed = model(**inputs, use_cache=False).logits[:, -1, :].float()
            change = (changed - baseline).abs().max().item()
            report['hook_checks'].append(dict(name='final_prediction_positive_control',
                passed=bool(torch.isfinite(changed).all()) and change > 1e-4, max_logit_change=change))
            report['hook_checks'].append(dict(name='all_hooks_removed',
                passed=all(not module._forward_hooks for module in model.modules())))
            save()
            if not all(check['passed'] for check in report['hook_checks']):
                raise AssertionError('Actual Aya hook validation failed')
            del baseline, observed, readout, changed, captured
            report['status'] = 'generating'
            save()
        prompts = [
            ('en', 'What is the capital of France? Answer with only the city name.'),
            ('he', 'מהי עיר הבירה של צרפת? יש להשיב רק בשם העיר.'),
            ('ar', 'ما عاصمة فرنسا؟ أجب باسم المدينة فقط.'),
        ]
        for language, prompt in prompts:
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            ids = output[0, inputs['input_ids'].shape[-1]:].tolist()
            elapsed = time.perf_counter() - started
            report['examples'].append(dict(language=language, prompt=prompt,
                input_ids=inputs['input_ids'][0].tolist(), generated_ids=ids,
                output=tokenizer.decode(ids, skip_special_tokens=True), seconds=elapsed,
                tokens_per_second=len(ids) / elapsed, human_review='pending'))
            save()
            print(json.dumps(report['examples'][-1], ensure_ascii=False), flush=True)
        report['status'] = 'inference_completed_native_review_pending'
        save()
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
