"""Bounded native Qwen3.8 reasoning generation followed by label scoring.

Generated token IDs are authoritative: decoded text is an archive, never a
source for retokenizing the scoring prefix. A capped or early-EOS trace is
explicitly force-closed; every input receives a scoring prefix.
"""
from __future__ import annotations

import time

THINK_END_ID = 248069
NEWLINE_ID = 271
LABEL_IDS = [15, 16]
OFFICIAL_EOS_IDS = [248046, 248044]
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}


def validate_scoring_suffix(tokenizer):
    """Verify the boundary without retokenizing noncanonical generated BPE IDs."""
    suffix = '</think>\n\n'
    if list(tokenizer.encode(suffix, add_special_tokens=False)) != [THINK_END_ID, NEWLINE_ID]:
        raise ValueError('expected </think> plus newline tokens [248069, 271]')
    for digit, token in zip(('0', '1'), LABEL_IDS):
        if list(tokenizer.encode(suffix + digit, add_special_tokens=False)) != [THINK_END_ID, NEWLINE_ID, token]:
            raise ValueError(f'label {digit} must preserve suffix and append exactly token {token}')
    return list(LABEL_IDS)


def generate_reasoning_batch(model, tokenizer, prompt_ids, device, max_new_tokens, seed, progress=None):
    """Return per-row reasoning IDs, stop metadata, and score-ready prefix IDs.

``generated_ids`` includes the first natural stop token and excludes subsequent
batch padding. ``reasoning_ids`` excludes that stop. ``scoring_prefix_ids`` is
the unpadded original prompt + reasoning IDs + [248069, 271]. The caller scores
``label_ids`` with its ordinary forward-pass scoring function. Seeds reproduce
the same batch composition/order; changing batching can change sampled traces.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    label_ids = validate_scoring_suffix(tokenizer)
    prompts = [list(ids) for ids in prompt_ids]
    if not prompts or any(not ids or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids) for ids in prompts):
        raise ValueError('nonempty prompts of nonnegative integer token IDs are required')
    if model.training:
        raise ValueError('reasoning generation requires model.eval()')
    if any(p.requires_grad or (p.is_floating_point() and p.dtype != torch.bfloat16) for p in model.parameters()):
        raise ValueError('reasoning generation requires frozen BF16 model parameters')
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError('a pad_token_id is required for left-padded generation')
    stops = {THINK_END_ID, *OFFICIAL_EOS_IDS, int(pad_id)}
    for value in (getattr(tokenizer, 'eos_token_id', None),
                  getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)):
        if value is not None:
            stops.update(value if isinstance(value, (list, tuple)) else [value])
    stops = sorted(int(value) for value in stops)
    width = max(map(len, prompts))
    target = torch.device(device)
    inputs = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=target)
    attention = torch.zeros_like(inputs)
    for row, ids in enumerate(prompts):
        inputs[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=target)
        attention[row, -len(ids):] = 1
    if progress:
        progress({'event': 'reasoning_started', 'batch_rows': len(prompts),
                  'max_new_tokens': max_new_tokens, 'seed': seed})
    devices = [] if target.type != 'cuda' else [target.index if target.index is not None else torch.cuda.current_device()]
    started = time.monotonic()
    class Heartbeat(StoppingCriteria):
        last_update = started

        def __call__(self, input_ids, scores, **kwargs):
            now = time.monotonic()
            if progress and now - self.last_update >= 30:
                progress({'event': 'reasoning_progress', 'batch_rows': len(prompts),
                          'generated_steps': input_ids.shape[1] - width,
                          'max_new_tokens': max_new_tokens, 'generation_seconds': now - started})
                self.last_update = now
            return False
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        sequences = model.generate(
            input_ids=inputs, attention_mask=attention, max_new_tokens=max_new_tokens,
            eos_token_id=stops, pad_token_id=pad_id, use_cache=True, logits_to_keep=1,
            output_scores=False, output_logits=False, output_attentions=False,
            output_hidden_states=False, return_dict_in_generate=False,
            num_beams=1, num_return_sequences=1, min_new_tokens=0, min_length=0,
            forced_eos_token_id=None, stopping_criteria=StoppingCriteriaList([Heartbeat()]), **SAMPLING)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != len(prompts):
        raise ValueError('native generate must return one token-ID tensor row per input')
    if not width < sequences.shape[1] <= width + max_new_tokens or not torch.equal(sequences[:, :width], inputs):
        raise ValueError('native generate changed the padded prompt or returned an invalid generated length')
    generated_rows = sequences[:, width:].detach().cpu().tolist()
    elapsed = time.monotonic() - started
    del sequences, inputs, attention  # no cache or per-step vocabulary scores are retained
    records = []
    for prompt, row in zip(prompts, generated_rows):
        end = next((i for i, token in enumerate(row) if token in stops), None)
        if end is None:
            if len(row) != max_new_tokens:
                raise ValueError('generation ended before its budget without a recognized stop token')
            generated, reasoning, stop_token, reason = row, row, None, 'token_cap'
        else:
            generated, reasoning, stop_token = row[:end + 1], row[:end], row[end]
            reason = 'thinking_completed' if stop_token == THINK_END_ID else 'early_eos'
        completed = reason == 'thinking_completed'
        records.append({
            'prompt_ids': prompt, 'generated_ids': generated, 'reasoning_ids': reasoning,
            'generated_token_count': len(generated), 'reasoning_token_count': len(reasoning),
            'stop_reason': reason, 'stop_token_id': stop_token,
            'thinking_completed': completed, 'forced_close': not completed,
            'forced_close_reason': None if completed else reason,
            'scoring_prefix_ids': prompt + reasoning + [THINK_END_ID, NEWLINE_ID],
            'label_ids': list(label_ids),
            'reasoning_text': tokenizer.decode(reasoning, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'max_new_tokens': max_new_tokens, 'batch_seed': seed,
            'batch_generation_seconds': elapsed,
        })
    if progress:
        progress({'event': 'reasoning_finished', 'batch_rows': len(records),
                  'thinking_completed': sum(r['thinking_completed'] for r in records),
                  'forced_close': sum(r['forced_close'] for r in records),
                  'generated_tokens': sum(r['generated_token_count'] for r in records),
                  'generation_seconds': elapsed})
    return records
