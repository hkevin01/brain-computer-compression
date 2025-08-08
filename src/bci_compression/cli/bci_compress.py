#!/usr/bin/env python3
"""Command-line interface for BCI Compression Toolkit.

Commands:
  list               List available compressors (plugins)
  compress           Compress an input file
  decompress         Decompress a compressed file
  stream             Stream compress a file or simulated source

Common Options:
  --plugin NAME      Compressor plugin / algorithm name
  --device {cpu,cuda}
  --quality FLOAT    Quality level 0-1
  --metrics          Emit metrics JSON to stdout
  --json             Output machine-readable JSON summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from bci_compression import __version__
from bci_compression.algorithms import FEATURES
from bci_compression.core import create_compressor
from bci_compression.data_acquisition import iter_mmap_chunks


def list_compressors() -> Dict[str, Any]:
    return dict(FEATURES)


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == '.npy':
        arr: np.ndarray = np.load(p)  # type: ignore[assignment]
        return arr
    elif p.suffix == '.npz':
        with np.load(p) as z:
            key = list(z.keys())[0]
            arr = z[key]  # type: ignore[index]
            return np.asarray(arr)
    else:
        raise ValueError(f'Unsupported input format: {p.suffix}')


def cmd_list(_args: argparse.Namespace) -> int:
    feats = list_compressors()
    print('Available feature flags:')
    for k, v in feats.items():
        print(f'  {k}: {"yes" if v else "no"}')
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    arr = load_array(args.input)
    compressor = create_compressor(args.plugin, config=None)
    t0 = time.perf_counter()
    data, meta = compressor.compress(arr)
    dt = time.perf_counter() - t0
    out_path = Path(args.output)
    out_path.write_bytes(data)
    if args.metrics:
        meta['wall_time_s'] = dt
        meta['input_shape'] = list(arr.shape)
        meta['input_dtype'] = str(arr.dtype)
        print(json.dumps(meta, indent=2 if not args.json else None))
    else:
        print(f'Compressed {arr.shape} -> {len(data)} bytes in {dt * 1000:.2f} ms; ratio {meta.get("compression_ratio", 0):.2f}x')
    return 0


def cmd_decompress(args: argparse.Namespace) -> int:
    comp_bytes = Path(args.input).read_bytes()
    # Need metadata; minimal stub expects none; real implementations require sidecar JSON
    metadata: Dict[str, Any] = {}
    compressor = create_compressor(args.plugin, config=None)
    t0 = time.perf_counter()
    arr = compressor.decompress(comp_bytes, metadata)
    dt = time.perf_counter() - t0
    if args.output:
        np.save(args.output, arr)
    if args.metrics:
        print(json.dumps({'shape': list(arr.shape), 'dtype': str(arr.dtype), 'wall_time_s': dt}, indent=2 if not args.json else None))
    else:
        print(f'Decompressed to shape {arr.shape} in {dt * 1000:.2f} ms')
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    compressor = create_compressor(args.plugin, config=None)
    total_in = 0
    total_out = 0
    t_start = time.perf_counter()
    for chunk in iter_mmap_chunks(args.input, chunk_samples=args.chunk, dataset=args.dataset):
        c_bytes, meta = compressor.stream_chunk(chunk)
        total_in += chunk.nbytes
        total_out += len(c_bytes)
        if args.verbose:
            ratio = chunk.nbytes / max(len(c_bytes), 1)
            print(f'Chunk {chunk.shape} -> {len(c_bytes)} bytes ({ratio:.2f}x)')
    elapsed = time.perf_counter() - t_start
    overall_ratio = total_in / max(total_out, 1)
    summary = {
        'chunks_processed': True,
        'input_megabytes': total_in / 1e6,
        'compressed_megabytes': total_out / 1e6,
        'compression_ratio': overall_ratio,
        'throughput_mb_s': (total_in / 1e6) / elapsed if elapsed > 0 else 0.0,
        'wall_time_s': elapsed,
    }
    print(json.dumps(summary, indent=2 if not args.json else None) if args.metrics else f'Stream ratio {overall_ratio:.2f}x in {elapsed:.2f}s')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='bci-compress', description='BCI Compression Toolkit CLI')
    p.add_argument('--version', action='version', version=f'bci-compression {__version__}')
    sub = p.add_subparsers(dest='command', required=True)

    sp_list = sub.add_parser('list', help='List available compressors')
    sp_list.set_defaults(func=cmd_list)

    sp_comp = sub.add_parser('compress', help='Compress an input array file (.npy/.npz)')
    sp_comp.add_argument('-i', '--input', required=True)
    sp_comp.add_argument('-o', '--output', required=True)
    sp_comp.add_argument('--plugin', default='adaptive_lz')
    sp_comp.add_argument('--quality', type=float, default=0.8)
    sp_comp.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    sp_comp.add_argument('--metrics', action='store_true')
    sp_comp.add_argument('--json', action='store_true')
    sp_comp.set_defaults(func=cmd_compress)

    sp_decomp = sub.add_parser('decompress', help='Decompress a file produced by compress')
    sp_decomp.add_argument('-i', '--input', required=True)
    sp_decomp.add_argument('-o', '--output', required=False)
    sp_decomp.add_argument('--plugin', default='adaptive_lz')
    sp_decomp.add_argument('--metrics', action='store_true')
    sp_decomp.add_argument('--json', action='store_true')
    sp_decomp.set_defaults(func=cmd_decompress)

    sp_stream = sub.add_parser('stream', help='Stream-compress large file via memory-mapped chunks')
    sp_stream.add_argument('-i', '--input', required=True)
    sp_stream.add_argument('--plugin', default='adaptive_lz')
    sp_stream.add_argument('--chunk', type=int, default=30_000)
    sp_stream.add_argument('--dataset', type=str, default='data')
    sp_stream.add_argument('--metrics', action='store_true')
    sp_stream.add_argument('--json', action='store_true')
    sp_stream.add_argument('-v', '--verbose', action='store_true')
    sp_stream.set_defaults(func=cmd_stream)
    return p


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc: int = args.func(args)
    return int(rc)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
