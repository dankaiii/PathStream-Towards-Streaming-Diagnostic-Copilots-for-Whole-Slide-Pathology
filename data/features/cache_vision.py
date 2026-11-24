"""Cache all four pre-merger ViT outputs with independent GPU workers."""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from .paths import resolve_reference
from .pipeline import BatchProducer, SessionWriter
from data.dataset import split_entries
from train.config import add_data_arguments, load_config, read_json, write_json

PACKAGE = 'data.features'


def load_vision(model_path, device):
    raw = read_json(Path(model_path) / 'config.json')
    config = Qwen3VLVisionConfig(**raw['vision_config'])
    config._attn_implementation = 'flash_attention_2'
    vision = Qwen3VLVisionModel(config)
    # Identity returns exactly the merger input, before either kind of LayerNorm.
    vision.merger = torch.nn.Identity()
    vision.deepstack_merger_list = torch.nn.ModuleList(
        [torch.nn.Identity() for _ in config.deepstack_visual_indexes])
    shards = defaultdict(list)
    for name, shard in read_json(Path(model_path) / 'model.safetensors.index.json')['weight_map'].items():
        if name.startswith('model.visual.') and not name.startswith(
                ('model.visual.merger.', 'model.visual.deepstack_merger_list.')):
            shards[shard].append(name)
    state = {}
    for shard, names in shards.items():
        with safe_open(str(Path(model_path) / shard), framework='pt', device='cpu') as stream:
            for name in names:
                state[name.removeprefix('model.visual.')] = stream.get_tensor(name).to(torch.bfloat16)
    vision.load_state_dict(state, assign=True)
    return vision.eval().requires_grad_(False).to(device)


def training_manifests(config):
    records = []
    for settings in config['stages'].values():
        directory = Path(config['dataset_root']) / settings['data_dir']
        for split in (settings['train_split'], settings['validation_split']):
            rows, entries = split_entries(directory, config['model_key'], split)
            for entry in entries:
                row = rows[entry['row_index']]
                manifest = resolve_reference(row['metadata']['frame_manifest'], directory, config['path_map'])
                records.append({'id': row['id'], 'manifest': str(manifest),
                                'frame_count': row['frame_count']})
    return records


def make_workloads(records, world):
    sessions = {}
    for record in records:
        job = sessions.setdefault(record['id'], {'id': record['id'], 'manifests': [],
                                  'frame_count': 0})
        if record['manifest'] not in job['manifests']:
            job['manifests'].append(record['manifest'])
            job['frame_count'] += record['frame_count']
    assignments = [[] for _ in range(world)]
    loads = [0] * world
    for job in sorted(sessions.values(), key=lambda job: job['frame_count'], reverse=True):
        rank = min(range(world), key=lambda i: loads[i])
        assignments[rank].append(job)
        loads[rank] += job['frame_count']
    return assignments


def controller(args, config, root):
    if args.manifest_list is None:
        records = training_manifests(config)
    else:
        records = []
        for record in map(json.loads, args.manifest_list.read_text().splitlines()):
            manifest = resolve_reference(record['manifest'], args.manifest_list.resolve().parent, config['path_map'])
            document = read_json(manifest)
            records.append({**record, 'manifest': str(manifest), 'frame_count': len(document['frames'])})
    assignments = make_workloads(records, args.workers)
    root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for rank, jobs in enumerate(assignments):
        write_json(args.log_dir / f'workload_rank{rank}.json', jobs)
    write_json(args.log_dir / 'workload_summary.json', {
        'sessions': sum(map(len, assignments)),
        'frames': sum(job['frame_count'] for jobs in assignments for job in jobs),
        'workers': [{'rank': rank, 'sessions': len(jobs),
                     'frames': sum(job['frame_count'] for job in jobs)}
                    for rank, jobs in enumerate(assignments)],
        'source_frame_policy': 'original_manifest_indices_and_viewports',
        'max_read_pixels': config['cache_max_read_pixels'],
    })
    resolved_config = args.log_dir / 'cache_config.json'
    write_json(resolved_config, config)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=' + str(args.workers), '-m', PACKAGE + '.cache_vision',
               '--config', str(resolved_config), '--log-dir', str(args.log_dir)]
    subprocess.run(command, check=True)
    excluded = []
    summaries = []
    for rank in range(args.workers):
        with (args.log_dir / f'skipped_sessions_rank{rank}.jsonl').open() as stream:
            excluded.extend(map(json.loads, stream))
        summaries.append(read_json(args.log_dir / f'summary_rank{rank}.json'))
    write_json(root / 'excluded_sessions.json', {
        'policy': 'exclude_whole_session_preserve_sft_alignment',
        'max_read_pixels': config['cache_max_read_pixels'],
        'session_ids': sorted(record['id'] for record in excluded), 'sessions': excluded})
    summary = {'status': 'completed', 'feature_root': str(root), 'logs': str(args.log_dir),
               **{key: sum(row[key] for row in summaries)
                  for key in ('sessions', 'written_sessions', 'excluded_sessions', 'unique_frames', 'bytes')}}
    write_json(root / 'cache_summary.json', summary)
    write_json(args.log_dir / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def worker(args, config, root):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    jobs = read_json(args.log_dir / f'workload_rank{rank}.json')
    vision = load_vision(config['model'], device)
    grouped_width = vision.config.hidden_size * vision.config.spatial_merge_size ** 2
    writer = SessionWriter(root, args.log_dir, rank, len(jobs), config)
    producer = BatchProducer(jobs, config)
    compute = torch.cuda.current_stream(device)
    upload, download = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    pending = deque()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for batch in producer:
            if len(pending) >= config['cache_write_backlog']:
                pending.popleft().result()
            if batch['kind'] == 'excluded':
                pending.append(pool.submit(writer.exclude, batch['session']))
                continue
            with torch.cuda.stream(upload):
                pixels = batch['pixel_values'].to(device, non_blocking=True)
                grid = batch['image_grid_thw'].to(device, non_blocking=True)
                ready = upload.record_event()
            compute.wait_event(ready)
            pixels.record_stream(compute)
            grid.record_stream(compute)
            with torch.inference_mode():
                last, deep = vision(pixels, grid_thw=grid)
                packed = torch.stack([last.reshape(-1, grouped_width),
                                      *[layer.reshape(-1, grouped_width) for layer in deep]])
                done = compute.record_event()
            features = torch.empty_like(packed, device='cpu', pin_memory=True)
            with torch.cuda.stream(download):
                download.wait_event(done)
                features.copy_(packed, non_blocking=True)
                copied = download.record_event()
            packed.record_stream(download)
            pending.append(pool.submit(writer.consume, features, copied, batch, packed))
        for future in pending:
            future.result()
    writer.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument('--manifest-list', type=Path)
    parser.add_argument('--frame-reader-python', help='Python interpreter with OpenSlide and Pillow')
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    config = load_config(args)
    root = Path(config['feature_root'])
    if args.controller:
        controller(args, config, root)
    else:
        worker(args, config, root)


if __name__ == '__main__':
    main()
