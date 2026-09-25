"""Fresh Qwen27B QLoRA classifiers on canonical OC exports.

Independent copy of the tested KIP optimization loop, using fixed OC splits.
Only validation loss selects checkpoints and stops training; no test probing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.experiments.kip_encoder_training import (
    DynamicPadding, EarlyStopping, TokenizedDataset, atomic_json, atomic_checkpoint,
    metric_values, read_split, restore_trainable, seed_all, sha256, subset_indices,
    trainable_state, validation_threshold, rng_state, restore_rng,
)
from src.experiments.kip_llama_training import accumulation_samples, classifier_loss, evaluate

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('qwen27b_qlora',)
FRACTIONS = (0.01, 0.10, 0.30, 0.50, 0.75, 1.0)


def diagnostic_test_evaluation(model, loader, device, generator):
    """A budget-stopping diagnostic must not alter subsequent optimization RNG."""
    state, was_training = rng_state(generator), model.training
    try:
        return evaluate(model, loader, device)
    finally:
        restore_rng(state, generator)
        model.train(was_training)


def selected_indices(labels, fraction, seed, *, smoke=False, split='train'):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must be finite and in (0, 1]')
    limit = int(len(labels) * fraction) if split == 'train' else len(labels)
    if smoke:
        limit = min(limit, 256 if split == 'train' else 128)
    return subset_indices(labels, limit, seed)


def model_provenance(path):
    """Record pinned HF identities and hash small config files without rehashing 54GB per fit."""
    index = path / 'model.safetensors.index.json'
    weights = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    records = {}
    for name in weights:
        if Path(name).name != name:
            raise ValueError('invalid checkpoint shard name')
        shard = path / name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        records[name] = {'bytes': shard.stat().st_size,
                         'expected_hf_blob_sha256': shard.resolve().name,
                         'verification': 'pinned HF cache blob identity and size; payload hash in queue evidence'}
    return {'snapshot': str(path.resolve()), 'revision': path.name,
            'metadata_sha256': {p.name: sha256(p) for p in sorted(path.iterdir())
                               if p.is_file() and p.suffix in {'.json', '.txt', '.jinja'}},
            'weights': records}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, default='qwen27b_qlora')
    parser.add_argument('--tag', choices=('ocdet', 'ocnoise'), default='ocdet')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'results/qwen25_14b_fewshot_20260918/data/oc_corrected')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=9950)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--progress-steps', type=int, default=20)
    parser.add_argument('--optimizer-offload', choices=['none', 'cpu'], default='none')
    parser.add_argument('--smoke', action='store_true')

    args = parser.parse_args(argv)
    args.stop_on_perfect_test = False
    if args.fraction != 1.0:
        parser.error('OC uses the complete training split; fraction must be 1')
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 20)
    for name in ('batch_size', 'gradient_accumulation_steps', 'eval_batch_size', 'epochs',
                 'patience', 'threads', 'progress_steps'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if not math.isfinite(args.fraction) or not 0 < args.fraction <= 1:
        parser.error('fraction must be in (0,1]')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be positive and finite')
    if args.optimizer_offload != 'none' and args.arm != 'llama8b_full':
        parser.error('CPU optimizer offload is only for full tuning')
    return args


def train(args):
    from src.experiments.kip_efficiency_models import load_classifier

    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'new fit requires empty output directory: {output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    seed_all(args.seed)
    device = torch.device('cuda')
    base = {'arm': args.arm, 'fraction': args.fraction, 'seed': args.seed,
            'tag': args.tag, 'mode': 'ordered', 'is_smoke': args.smoke}

    def progress(state, **details):
        record = {**base, 'status': state, 'pid': os.getpid(),
                  'updated_at_unix': time.time(), 'wall_seconds': time.monotonic() - started,
                  **details}
        atomic_json(record, output / 'status.json')
        print(json.dumps({'event': state, **details}, allow_nan=False), flush=True)

    progress('loading_data')
    selected, provenance, rows = {}, {}, {}
    expected = {'train': 400000, 'val': 50000, 'test': 50000}
    for split in ('train', 'val', 'test'):
        sequences, labels, metadata = read_split(args.data_dir, args.tag, split)
        if len(labels) != expected[split]:
            raise ValueError(f'{split} requires {expected[split]} source rows')
        indices = selected_indices(labels, args.fraction, args.seed, smoke=args.smoke, split=split)
        selected[split] = ([sequences[i] for i in indices], labels[indices], indices)
        provenance[split] = metadata
        rows[split] = len(indices)
        np.save(output / f'{split}_indices.npy', indices, allow_pickle=False)
    config = {**base, 'schema_version': 1, 'model_source': model_provenance(args.model_path),
              'data': provenance, 'rows': rows, 'batch_size': args.batch_size,
              'gradient_accumulation_steps': args.gradient_accumulation_steps,
              'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
              'eval_batch_size': args.eval_batch_size, 'epochs': args.epochs,
              'patience': args.patience, 'lr': args.lr, 'weight_decay': 0.01,
              'warmup_ratio': 0.06, 'optimizer': 'AdamW', 'optimizer_state_dtype': 'float32',
              'optimizer_offload': args.optimizer_offload, 'precision': 'bf16_autocast',
              'gradient_checkpointing': True, 'max_length': 64,
              'prompt': 'Sequential events: ' + ' '.join(['<event>'] * 20),
              'loss': 'binary_outcome_cross_entropy_float32',
              'selection': 'minimum_validation_loss', 'threshold_selection': 'validation_f1',
              'stop_on_perfect_test': args.stop_on_perfect_test,
              'test_informed_stopping': args.stop_on_perfect_test and not args.smoke,
              'test_check_trigger': 'new_minimum_validation_loss_with_validation_auc_1',
              'sampling': 'stratified sklearn train_test_split; sorted original row indices; fixed seed; not necessarily nested',
              'fresh_pretrained_initialization': True,
              'checkpoint_policy': 'selected trainable adapter/head saved; full weights CPU RAM only; unfinished fit restarts',
              'index_sha256': {s: sha256(output / f'{s}_indices.npy') for s in selected}}
    atomic_json({'config': config, 'created_at_unix': time.time()}, output / 'manifest.json')
    progress('loading_model')
    model, tokenizer, model_metadata = load_classifier(args.arm, args.model_path, gradient_checkpointing=True)
    datasets = {split: TokenizedDataset(sequences, labels, tokenizer, 64)
                for split, (sequences, labels, _) in selected.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(data, batch_size=args.batch_size if split == 'train' else args.eval_batch_size,
                                shuffle=split == 'train', generator=generator if split == 'train' else None,
                                collate_fn=DynamicPadding(tokenizer.pad_token_id, 64),
                                num_workers=0, pin_memory=True)
               for split, data in datasets.items()}
    manifest = {'config': config, 'model': model_metadata, 'created_at_unix': time.time(),
                'token_lengths': {s: {'min': d.min_tokens, 'max': d.max_tokens} for s, d in datasets.items()},
                'environment': {p: importlib.metadata.version(p) for p in
                                ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate')},
                'gpu': torch.cuda.get_device_name(),
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(__file__), Path(__file__).with_name('kip_efficiency_models.py'),
                                   Path(__file__).with_name('kip_encoder_training.py'),
                                   Path(__file__).with_name('kip_llama_training.py')]}}
    atomic_json(manifest, output / 'manifest.json')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('trainable parameters must be nonempty and FP32')
    if args.optimizer_offload == 'cpu':
        from src.experiments.kip_cpu_offload import CPUOffloadAdamW
        optimizer = CPUOffloadAdamW(parameters, lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, foreach=False)
    total_steps = args.epochs * math.ceil(len(loaders['train']) / args.gradient_accumulation_steps)
    warmup_steps = int(0.06 * total_steps)

    def schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper, best_state, history, optimizer_steps = EarlyStopping(args.patience), None, [], 0
    perfect_test, perfect_validation = None, None
    stop_reason = 'max_epochs'
    training_started = time.monotonic()
    progress('ready', rows=rows, trainable_parameters=sum(p.numel() for p in parameters))
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = torch.zeros((), device=device, dtype=torch.float64), 0
        progress('training', epoch=epoch, batch=0, batches=len(loaders['train']))
        for index, batch in enumerate(loaders['train']):
            stop, group_samples = accumulation_samples(index, len(loaders['train']), args.batch_size,
                                                       rows['train'], args.gradient_accumulation_steps)
            loss, _ = classifier_loss(model, batch, device, 'bf16')
            count = len(batch['labels'])
            (loss * count / group_samples).backward()
            loss_sum += loss.detach().double() * count
            seen += count
            if index + 1 == stop:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (index + 1) % args.progress_steps == 0 or index + 1 == len(loaders['train']):
                average = loss_sum.item() / seen
                if not math.isfinite(average):
                    raise ValueError('non-finite training loss')
                progress('training', epoch=epoch, batch=index + 1, batches=len(loaders['train']),
                         train_loss=average, optimizer_steps=optimizer_steps,
                         examples_per_second=seen / (time.monotonic() - epoch_started))
        progress('validating', epoch=epoch)
        val_loss, val_probabilities, val_labels = evaluate(model, loaders['val'], device)
        improved = stopper.update(val_loss, epoch)
        if improved:
            if best_state is None:
                best_state = trainable_state(model)
            else:
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        best_state[name].copy_(parameter.detach())
            if args.arm != 'llama8b_full':
                needed = sum(t.numel() * t.element_size() for t in best_state.values()) + 64 * 1024**2
                if shutil.disk_usage(output).free < needed:
                    raise OSError('not enough disk space for selected adapter checkpoint')
                atomic_checkpoint({'arm': args.arm, 'epoch': epoch, 'state_dict': best_state}, output / 'best_trainable.pt')
        record = {'epoch': epoch, 'train_loss': loss_sum.item() / seen, 'val_loss': val_loss,
                  'val_auc': metric_values(val_labels, val_probabilities, 0.5)['auc'],
                  'improved': improved, 'best_epoch': stopper.best_epoch, 'bad_epochs': stopper.bad_epochs,
                  'optimizer_steps': optimizer_steps, 'learning_rate': scheduler.get_last_lr()[0],
                  'epoch_seconds': time.monotonic() - epoch_started}
        history.append(record)
        with (output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        progress('epoch_complete', **record)
        if args.stop_on_perfect_test and not args.smoke and improved and record['val_auc'] == 1.0:
            progress('checking_test_auc_stop_rule', epoch=epoch)
            probe = diagnostic_test_evaluation(model, loaders['test'], device, generator)
            probe_auc = metric_values(probe[2], probe[1], 0.5)['auc']
            proof = output / f'test_check_epoch_{epoch:03d}.csv'
            pd.DataFrame({'row_index': selected['test'][2], 'label': probe[2],
                          'probability': probe[1]}).to_csv(proof, index=False)
            check = {'epoch': epoch, 'checkpoint_epoch': stopper.best_epoch,
                     'trigger': config['test_check_trigger'], 'validation_loss': val_loss,
                     'validation_auc': record['val_auc'], 'test_rows': len(probe[2]),
                     'test_auc': probe_auc, 'exact_test_auc_1': probe_auc == 1.0,
                     'predictions': proof.name, 'predictions_sha256': sha256(proof),
                     'test_informed_stopping': True}
            with (output / 'test_checks.jsonl').open('a') as handle:
                handle.write(json.dumps(check, allow_nan=False) + '\n')
            progress('test_auc_stop_check_complete', **check)
            if probe_auc == 1.0:
                perfect_test = probe
                perfect_validation = (val_loss, val_probabilities, val_labels)
                stop_reason = 'test_auc_1'
                break
        if stopper.stopped:
            stop_reason = 'validation_patience'
            break
    if best_state is None:
        raise RuntimeError('no selected model')
    training_seconds = time.monotonic() - training_started
    restore_trainable(model, best_state)
    del best_state, optimizer, scheduler
    torch.cuda.empty_cache()
    progress('evaluating', best_epoch=stopper.best_epoch)
    val_loss, val_probabilities, val_labels = (perfect_validation if perfect_validation is not None
                                             else evaluate(model, loaders['val'], device))
    threshold = validation_threshold(val_labels, val_probabilities)
    test_loss, test_probabilities, test_labels = (perfect_test if perfect_test is not None
                                                else evaluate(model, loaders['test'], device))
    for split, probabilities, labels in [('val', val_probabilities, val_labels),
                                         ('test', test_probabilities, test_labels)]:
        pd.DataFrame({'row_index': selected[split][2], 'label': labels, 'probability': probabilities,
                      'prediction': (probabilities >= threshold).astype(int),
                      'prediction_at_0_5': (probabilities >= 0.5).astype(int)}).to_csv(
                          output / f'{split}_predictions.csv', index=False)
    tokenizer.save_pretrained(output / 'tokenizer')
    metrics = {**base, 'status': 'completed', 'schema_version': 1, 'rows': rows,
               'best_epoch': stopper.best_epoch, 'epochs_completed': history[-1]['epoch'],
               'early_stopped': stopper.stopped or stop_reason == 'test_auc_1', 'selected_by': 'validation_loss',
               'stop_reason': stop_reason,
               'stopping_rule': 'validation_gated_test_auc_1' if args.stop_on_perfect_test else 'validation_loss_patience',
               'test_informed_stopping': args.stop_on_perfect_test and not args.smoke,
               'test_check_trigger': config['test_check_trigger'],
               'threshold_selected_on': 'validation', 'training_seconds': training_seconds,
               'wall_seconds': time.monotonic() - started,
               'validation': {'loss': val_loss, **metric_values(val_labels, val_probabilities, threshold)},
               'validation_at_0_5': metric_values(val_labels, val_probabilities, 0.5),
               'test': {'loss': test_loss, **metric_values(test_labels, test_probabilities, threshold)},
               'test_at_0_5': metric_values(test_labels, test_probabilities, 0.5),
               'selected_weights_saved': args.arm != 'llama8b_full',
               'checkpoint_policy': config['checkpoint_policy']}
    artifacts = [p for p in output.rglob('*') if p.is_file() and p.name != 'status.json']
    metrics['artifact_sha256'] = {str(p.relative_to(output)): sha256(p) for p in artifacts}
    atomic_json(metrics, output / 'metrics.json')
    progress('completed', best_epoch=stopper.best_epoch, test_auc=metrics['test']['auc'])
    return metrics


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except Exception as error:
        if args.output_dir.is_dir() and not isinstance(error, FileExistsError):
            atomic_json({'status': 'failed', 'pid': os.getpid(), 'updated_at_unix': time.time(),
                         'error_type': type(error).__name__, 'error': str(error)}, args.output_dir / 'status.json')
        raise


if __name__ == '__main__':
    main()
