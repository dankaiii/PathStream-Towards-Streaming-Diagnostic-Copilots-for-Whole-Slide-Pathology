"""CPU readers, cross-session batching and asynchronous session-file writing."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Lock, Thread, local
import time

from PIL import Image
import torch

from data.conversation import make_processor
from train.config import write_json


class SessionReaders:
    def __init__(self, config):
        self.config = config
        self.local = local()
        self.processes = []
        self.lock = Lock()

    def read(self, job):
        if not hasattr(self.local, 'reader'):
            command = [self.config['frame_reader_python'] or sys.executable, str(Path(__file__).with_name('read_frames.py')),
                       '--server', '--max-read-pixels', str(self.config['cache_max_read_pixels'])]
            for old, new in self.config['path_map']:
                command.extend(['--path-map', old, new])
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            self.local.reader = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            self.local.processor = make_processor(self.config['model'], self.config['max_pixels']).image_processor
            with self.lock:
                self.processes.append(self.local.reader)
        reader = self.local.reader
        reader.stdin.write((json.dumps(job) + '\n').encode())
        reader.stdin.flush()
        header = json.loads(reader.stdout.readline())
        if header['status'] == 'excluded':
            return header
        width, height = header['width'], header['height']
        images = [Image.frombytes('RGB', (width, height), reader.stdout.read(width * height * 3))
                  for _ in header['source_frame_record_indices']]
        inputs = self.local.processor(images=images, return_tensors='pt')
        return {**header, 'pixel_values': inputs['pixel_values'].to(torch.bfloat16),
                'image_grid_thw': inputs['image_grid_thw']}

    def close(self):
        for process in self.processes:
            process.stdin.close()
        for process in self.processes:
            process.wait()
            process.stdout.close()


def prepared_sessions(jobs, config):
    readers = SessionReaders(config)
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=config['cache_read_workers']) as pool:
        pending = set()
        for _ in range(min(len(jobs), config['cache_prefetch_sessions'])):
            pending.add(pool.submit(readers.read, next(iterator)))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(readers.read, job))
    readers.close()


def packed_batches(jobs, config):
    parts, pixels, grids, frame_count = [], [], [], 0

    def pack():
        return {'kind': 'batch', 'parts': list(parts),
                'pixel_values': torch.cat(pixels).pin_memory(),
                'image_grid_thw': torch.cat(grids).pin_memory()}

    for session in prepared_sessions(jobs, config):
        if session['status'] == 'excluded':
            yield {'kind': 'excluded', 'session': session}
            continue
        indices, grid = session['source_frame_record_indices'], session['image_grid_thw']
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), grid.prod(-1).cumsum(0)))
        begin = 0
        while begin < len(indices):
            count = min(config['cache_frame_batch_size'] - frame_count, len(indices) - begin)
            end = begin + count
            pixels.append(session['pixel_values'][int(offsets[begin]):int(offsets[end])])
            grids.append(grid[begin:end])
            parts.append({'id': session['id'], 'offset': begin, 'total_frames': len(indices),
                          'source_frame_record_indices': indices[begin:end],
                          'source_manifests': session['source_manifests'], 'image_grid_thw': grid[begin:end]})
            frame_count += count
            begin = end
            if frame_count == config['cache_frame_batch_size']:
                yield pack()
                parts, pixels, grids, frame_count = [], [], [], 0
    if frame_count:
        yield pack()


class BatchProducer:
    def __init__(self, jobs, config):
        self.queue = Queue(maxsize=config['cache_prefetch_batches'])
        self.thread = Thread(target=self.run, args=(jobs, config), daemon=True)
        self.thread.start()

    def run(self, jobs, config):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        try:
            for batch in packed_batches(jobs, config):
                self.queue.put(batch)
        except BaseException as error:
            self.queue.put(error)
        finally:
            self.queue.put(None)

    def __iter__(self):
        while True:
            batch = self.queue.get()
            if isinstance(batch, BaseException):
                raise batch
            if batch is None:
                self.thread.join()
                return
            yield batch


class SessionWriter:
    def __init__(self, root, logs, rank, total_sessions, config):
        self.root, self.logs, self.rank, self.config = root, logs, rank, config
        self.total_sessions = total_sessions
        self.pending = {}
        self.completed = self.written = self.excluded = self.frames = self.bytes = 0
        self.started = time.monotonic()
        self.log = (logs / f'features_rank{rank}.jsonl').open('w')
        self.skip_log = (logs / f'skipped_sessions_rank{rank}.jsonl').open('w')

    def progress(self, sid, status):
        elapsed = time.monotonic() - self.started
        record = {'id': sid, 'status': status, 'completed_sessions': self.completed,
                  'total_sessions': self.total_sessions, 'written_sessions': self.written,
                  'excluded_sessions': self.excluded, 'unique_frames': self.frames,
                  'bytes': self.bytes, 'frames_per_second': self.frames / elapsed,
                  'elapsed_seconds': elapsed}
        self.log.write(json.dumps(record) + '\n')
        self.log.flush()
        write_json(self.logs / f'progress_rank{self.rank}.json', record)
        print(json.dumps({'rank': self.rank, **record}), flush=True)

    def exclude(self, session):
        record = {'id': session['id'], 'source_manifests': session['source_manifests'],
                  'policy': 'exclude_whole_session_preserve_sft_alignment', 'violations': session['violations']}
        self.skip_log.write(json.dumps(record) + '\n')
        self.skip_log.flush()
        self.completed += 1
        self.excluded += 1
        self.progress(session['id'], 'excluded')

    def consume(self, features, event, batch, gpu_reference):
        # CPU serialization starts only after this batch's asynchronous D2H copy.
        event.synchronize()
        cursor = 0
        for part in batch['parts']:
            count = int(part['image_grid_thw'].prod(-1).sum()) // 4
            current = {**part, 'features': features[0, cursor:cursor + count].clone(),
                       'deepstack_features': features[1:, cursor:cursor + count].clone()}
            cursor += count
            pieces = self.pending.setdefault(part['id'], [])
            pieces.append(current)
            if sum(len(piece['source_frame_record_indices']) for piece in pieces) != part['total_frames']:
                continue
            pieces.sort(key=lambda piece: piece['offset'])
            payload = {'features': torch.cat([piece['features'] for piece in pieces]),
                       'deepstack_features': torch.cat([piece['deepstack_features'] for piece in pieces], dim=1),
                       'image_grid_thw': torch.cat([piece['image_grid_thw'] for piece in pieces]),
                       'source_frame_record_indices': [index for piece in pieces for index in piece['source_frame_record_indices']],
                       'source_manifests': part['source_manifests'],
                       'layout': 'vit_premerger_source_frame_union_v1', 'feature_stage': 'vit_premerger',
                       'model': self.config['model'], 'max_pixels': self.config['max_pixels'],
                       'vit_hidden_size': 1024, 'grouped_width': 4096,
                       'deepstack_visual_indexes': [5, 11, 17],
                       'normalization': 'before_each_merger_layernorm', 'dtype': 'bfloat16',
                       'sequence_policy': 'unique_source_frame_record_index_manifest_order',
                       'max_read_pixels': self.config['cache_max_read_pixels']}
            destination = self.root / (part['id'] + '.pt')
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            del self.pending[part['id']]
            self.frames += part['total_frames']
            self.bytes += destination.stat().st_size
            self.completed += 1
            self.written += 1
            self.progress(part['id'], 'completed')

    def finish(self):
        self.log.close()
        self.skip_log.close()
        write_json(self.logs / f'summary_rank{self.rank}.json', {
            'status': 'completed', 'sessions': self.completed, 'written_sessions': self.written,
            'excluded_sessions': self.excluded, 'unique_frames': self.frames, 'bytes': self.bytes,
            'feature_root': str(self.root), 'elapsed_seconds': time.monotonic() - self.started})
