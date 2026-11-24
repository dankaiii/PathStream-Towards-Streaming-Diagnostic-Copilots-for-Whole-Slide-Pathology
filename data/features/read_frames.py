#!/usr/bin/env python3
"""Persistent RGB reader with a pre-decode pixel limit and original frame geometry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import openslide
from PIL import Image, ImageOps

from paths import resolve_reference
from wsi_viewport import render_wsi_viewport, viewport_read_spec


def read_manifest(path, path_map):
    document = json.loads(path.read_text())
    document['source_path'] = str(resolve_reference(document['source_path'], path.parent, path_map))
    return document


def source_video_images(document):
    width, height = document['width'], document['height']
    indices = sorted({frame['source_frame_index'] for frame in document['frames']})
    selected = '+'.join(f'eq(n,{index})' for index in indices)
    filters = (f"select='{selected}',"
               f'scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,'
               f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2',
               '-i', document['source_path'], '-an', '-vf', filters, '-filter_threads', '1',
               '-vsync', '0', '-frames:v', str(len(indices)), '-pix_fmt', 'rgb24',
               '-f', 'rawvideo', 'pipe:1']
    images = {}
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        for index in indices:
            images[index] = Image.frombytes('RGB', (width, height), process.stdout.read(width * height * 3))
        process.stdout.close()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    return images


def preflight(document, limit):
    violations = []
    width, height = document['width'], document['height']
    if width * height > limit:
        return [{'reason': 'render_size_exceeds_limit', 'width': width, 'height': height,
                 'pixels': width * height, 'limit': limit}]
    if document['source_type'] == 'video':
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height', '-of', 'json', document['source_path']]
        info = json.loads(subprocess.check_output(command))['streams'][0]
        pixels = info['width'] * info['height']
        if pixels > limit:
            violations.append({'reason': 'video_decode_size_exceeds_limit', **info,
                               'pixels': pixels, 'limit': limit})
    else:
        with openslide.OpenSlide(document['source_path']) as slide:
            for frame in document['frames']:
                spec = viewport_read_spec(slide, frame['viewport'], (width, height))
                if spec['pixels'] > limit:
                    violations.append({'reason': 'wsi_read_region_exceeds_limit',
                                       'source_frame_record_index': frame['source_frame_record_index'],
                                       'viewport': frame['viewport'], 'level': spec['level'],
                                       'read_size': spec['crop_size'], 'pixels': spec['pixels'], 'limit': limit})
    return violations


def iter_frames(document, limit):
    if document['source_type'] == 'video':
        images = source_video_images(document)
        for frame in document['frames']:
            yield images[frame['source_frame_index']]
    else:
        with openslide.OpenSlide(document['source_path']) as slide:
            for frame in document['frames']:
                image = render_wsi_viewport(slide, frame['viewport'],
                                            (document['width'], document['height']), max_read_pixels=limit)
                yield ImageOps.pad(image, (document['width'], document['height']),
                                   method=Image.Resampling.LANCZOS, color='black')


def serve(limit, path_map):
    for line in sys.stdin:
        request = json.loads(line)
        documents, indices, violations, seen = [], [], [], set()
        for manifest in request['manifests']:
            document = read_manifest(Path(manifest), path_map)
            frames = []
            for frame in document['frames']:
                index = frame['source_frame_record_index']
                if index not in seen:
                    seen.add(index)
                    indices.append(index)
                    frames.append(frame)
            if not frames:
                continue
            document = {**document, 'frames': frames}
            violations.extend({'manifest': manifest, 'source_path': document['source_path'], **item}
                              for item in preflight(document, limit))
            documents.append(document)
        header = {'id': request['id'], 'source_frame_record_indices': indices,
                  'source_manifests': request['manifests'],
                  'width': documents[0]['width'], 'height': documents[0]['height'],
                  'status': 'excluded' if violations else 'ready', 'violations': violations}
        sys.stdout.buffer.write((json.dumps(header) + '\n').encode())
        sys.stdout.buffer.flush()
        if violations:
            continue
        for document in documents:
            for image in iter_frames(document, limit):
                sys.stdout.buffer.write(image.tobytes())
        sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?')
    parser.add_argument('--frame-indices', type=int, nargs='+')
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--max-read-pixels', type=int, default=100_000_000)
    parser.add_argument('--path-map', nargs=2, action='append', default=[], metavar=('OLD', 'NEW'))
    args = parser.parse_args()
    if args.server:
        serve(args.max_read_pixels, args.path_map)
        return
    document = read_manifest(args.manifest, args.path_map)
    if args.frame_indices is not None:
        document['frames'] = [document['frames'][index] for index in args.frame_indices]
    violations = preflight(document, args.max_read_pixels)
    if violations:
        print(json.dumps({'status': 'excluded', 'manifest': str(args.manifest), 'violations': violations}), file=sys.stderr)
        raise SystemExit(2)
    for image in iter_frames(document, args.max_read_pixels):
        sys.stdout.buffer.write(image.tobytes())


if __name__ == '__main__':
    main()
