"""Bounded 50M supervised-token MiniMind pretraining, with archived run contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

PIN = '6fc918beb68a0d8c40452338df6319fe168014ba'
SOURCE = os.environ.get('MINIMIND_DATA_URL', 'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t_mini.jsonl')
CONTRACT = dict(seed=20260907, hidden_size=768, num_hidden_layers=8, use_moe=False,
                seq_len=340, micro_batch=8, accumulation=8, epochs=1,
                learning_rate=5e-4, grad_clip=1.0, dtype='float16',
                target_supervised_tokens=50_000_000, validation_samples=4096)


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(value, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(tokenizer, data):
    import requests
    manifest_path = data / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract'] != CONTRACT:
            raise RuntimeError('existing dataset contract differs')
        for name in ('train', 'validation'):
            if sha(data / (name + '.jsonl')) != manifest[name]['sha256']:
                raise RuntimeError('dataset hash mismatch: ' + name)
        return manifest
    data.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = {'train': 0, 'validation': 0}
    tokens = dict(counts)
    paths = {name: data / (name + '.jsonl') for name in counts}
    with requests.get(SOURCE, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with open(paths['train'], 'w', encoding='utf-8') as train, open(paths['validation'], 'w', encoding='utf-8') as val:
            for index, raw in enumerate(response.iter_lines()):
                if not raw:
                    continue
                row = json.loads(raw)
                text = row.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                if content_hash in seen:
                    continue
                seen.add(content_hash)
                ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=338).input_ids
                ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                supervised = sum(token != tokenizer.pad_token_id for token in ids[1:])
                name = 'validation' if counts['validation'] < CONTRACT['validation_samples'] else 'train'
                record = dict(text=text, source_index=index, content_sha256=content_hash, supervised_tokens=supervised)
                (val if name == 'validation' else train).write(json.dumps(record, ensure_ascii=False) + '\n')
                counts[name] += 1
                tokens[name] += supervised
                if name == 'train' and counts[name] % 10000 == 0:
                    print(f'DATA train_samples={counts[name]} supervised_tokens={tokens[name]}', flush=True)
                if tokens['train'] >= CONTRACT['target_supervised_tokens'] and counts['train'] % 64 == 0:
                    break
    if tokens['train'] < CONTRACT['target_supervised_tokens'] or counts['validation'] != 4096:
        raise RuntimeError('insufficient source data')
    manifest = dict(contract=CONTRACT, source=SOURCE, content_hash_deduplication=True,
                    train_validation_overlap=0)
    for name in counts:
        manifest[name] = dict(samples=counts[name], supervised_tokens=tokens[name], sha256=sha(paths[name]))
    write_json(manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    mini = repo / 'third_party/minimind'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=mini, text=True).strip() != PIN:
        raise RuntimeError('MiniMind commit mismatch')
    sys.path.insert(0, str(mini))
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed, get_lr
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(mini / 'model')
    manifest = prepare(tokenizer, run / 'data')
    if args.prepare_only:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    checkpoint_path = run / 'last.pt'
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError('existing run requires --resume')
    if args.resume and not checkpoint_path.exists():
        raise RuntimeError('no checkpoint to resume')
    setup_seed(CONTRACT['seed'])
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    train = PretrainDataset(str(run / 'data/train.jsonl'), tokenizer, max_length=340)
    val = PretrainDataset(str(run / 'data/validation.jsonl'), tokenizer, max_length=340)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(CONTRACT['seed'])).tolist()
    total_micro = len(train) // 8
    if total_micro % 8:
        raise RuntimeError('dataset not aligned to accumulation boundary')
    milestones = {8 * math.ceil(total_micro * fraction / 8) for fraction in (0.25, 0.5, 0.75, 1.0)}
    runtime = dict(contract=CONTRACT, manifest=manifest, minimind_commit=PIN,
                   parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
                   driver_sha256=sha(__file__), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   total_micro_batches=total_micro, total_optimizer_steps=total_micro//8,
                   milestone_micro_steps=sorted(milestones), run_id=run.name)
    write_json(runtime, run / 'config.json')
    start = 0
    tokens_seen = 0
    history = []
    if args.resume:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state['contract'] != CONTRACT or state['manifest'] != manifest:
            raise RuntimeError('resume contract mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, tokens_seen, history = state['micro_step'], state['tokens_seen'], state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUME micro_step={start} optimizer_step={start//8} tokens={tokens_seen}', flush=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=8, shuffle=False, num_workers=2)

    def validate():
        model.eval()
        summed, count = 0.0, 0
        with torch.inference_mode():
            for x, labels in val_loader:
                n = int((labels[:,1:] != -100).sum())
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = model(x.cuda(), labels=labels.cuda()).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('validation loss nonfinite')
                summed += float(loss) * n
                count += n
        model.train()
        return summed / count

    def snapshot(tag):
        model.eval()
        outputs = []
        with torch.inference_mode():
            for prompt in ['人工智能是', '中国的首都是', '一加一等于']:
                ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model.generate(ids, max_new_tokens=48, do_sample=False, use_cache=True,
                                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                outputs.append(dict(prompt=prompt, output=tokenizer.decode(output[0].tolist()),
                                    generation_config=dict(max_new_tokens=48, do_sample=False, use_cache=True)))
        write_json(outputs, run / (tag + '_generation.json'))
        model.train()

    def save(step):
        state = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     scheduler=dict(type='get_lr_cosine', total_micro_steps=total_micro, base_lr=5e-4, last_micro_step=step),
                     micro_step=step, global_step=step//8, epoch=0, tokens_seen=tokens_seen,
                     history=history, contract=CONTRACT, manifest=manifest, config=runtime,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        tick = time.monotonic()
        atomic_save(state, run / f'step_{step//8}.pt')
        atomic_save(state, checkpoint_path)
        print(f'CHECKPOINT optimizer_step={step//8} seconds={time.monotonic()-tick:.2f}', flush=True)

    if not args.resume:
        initial_val = validate()
        print(f'INITIAL validation_loss={initial_val:.6f}', flush=True)
        write_json(dict(loss=initial_val), run / 'initial_validation.json')
        snapshot('step_0')
    sampler = [order[i:i+8] for i in range(start*8, len(order), 8)]
    loader = torch.utils.data.DataLoader(train, batch_sampler=sampler, num_workers=2, pin_memory=True)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    window_loss, window_tokens = 0.0, 0
    for step, (x, labels) in enumerate(loader, start=start+1):
        n = int((labels[:,1:] != -100).sum())
        lr = get_lr(step, total_micro, 5e-4)
        for group in optimizer.param_groups:
            group['lr'] = lr
        with torch.amp.autocast('cuda', dtype=torch.float16):
            result = model(x.cuda(non_blocking=True), labels=labels.cuda(non_blocking=True))
            loss = result.loss + result.aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'loss nonfinite micro_step={step}')
        scaler.scale(loss / 8).backward()
        tokens_seen += n
        window_loss += float(loss.detach()) * n
        window_tokens += n
        del result, loss
        if step % 8 == 0:
            scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(grad):
                raise FloatingPointError(f'gradient nonfinite micro_step={step}')
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            record = dict(optimizer_step=step//8, micro_step=step, train_loss=window_loss/window_tokens,
                          tokens_seen=tokens_seen, lr=lr, grad_norm=grad, scaler_scale=scaler.get_scale())
            history.append(record)
            if step//8 <= 5 or step//8 % 50 == 0:
                print(f'TRAIN {json.dumps(record)} elapsed_seconds={time.monotonic()-tick:.1f}', flush=True)
            window_loss, window_tokens = 0.0, 0
            if step in milestones:
                record['validation_loss'] = validate()
                print(f'VALIDATION micro_step={step} loss={record["validation_loss"]:.6f}', flush=True)
                if step >= total_micro//2 and not (run / 'middle_generation.json').exists():
                    snapshot('middle')
                save(step)
    snapshot('final')
    write_json(dict(completed=True, tokens_seen=tokens_seen, optimizer_steps=total_micro//8,
                    history=history, peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    wall_seconds=time.monotonic()-tick), run / 'metrics.json')
    print('FORMAL_PRETRAIN_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
