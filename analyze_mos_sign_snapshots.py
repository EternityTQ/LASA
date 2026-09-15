#!/usr/bin/env python3
"""Offline CPU replay for MOS LASA-aware Sign diagnostic snapshots."""

import argparse
import gc
import glob
import json
import os
import time
from types import SimpleNamespace

from algorithms.engine.mos_sign_shadow import analyze_snapshot, load_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_dir', required=True,
                        help='Snapshot .pt file or directory containing snapshots')
    parser.add_argument('--output_dir', default=None,
                        help='CSV output directory (defaults beside the snapshot)')
    parser.add_argument('--shared_candidates', type=int, default=None,
                        help='Target shared candidate count (default: snapshot config or 10)')
    parser.add_argument('--lasa_boundary_evals', type=int, default=None,
                        help='LASA-faithful boundary evaluation budget (default: snapshot config or 10)')
    args = parser.parse_args()
    source = os.path.abspath(args.snapshot_dir)
    paths = ([source] if os.path.isfile(source) else sorted(glob.glob(
        os.path.join(source, 'diagnostic_snapshot_round_*.pt'))))
    if not paths:
        raise SystemExit(f'No diagnostic snapshots found at {source}')
    output_dir = os.path.abspath(args.output_dir or (
        os.path.dirname(source) if os.path.isfile(source) else source))
    os.makedirs(output_dir, exist_ok=True)
    for filename in ('sign_proxy_comparison.csv',
                     'sign_proxy_common_candidates.csv'):
        result = os.path.join(output_dir, filename)
        if os.path.exists(result):
            os.replace(result, result + '.previous')
    for path in paths:
        started = time.perf_counter()
        snapshot_size = os.path.getsize(path)
        print(f'[SIGN-SHADOW-OFFLINE] loading path={path} '
              f'size_bytes={snapshot_size}', flush=True)
        snapshot = load_snapshot(path)
        try:
            round_index = snapshot['round']
            config = dict(snapshot['diagnostic_config'])
            config['device'] = 'cpu'
            if args.shared_candidates is not None:
                config['mos_sign_shared_candidates'] = max(1, args.shared_candidates)
            if args.lasa_boundary_evals is not None:
                config['mos_lasa_boundary_evals'] = max(2, args.lasa_boundary_evals)
            analyze_snapshot(snapshot, SimpleNamespace(**config), output_dir)
            metrics = dict(getattr(analyze_snapshot, 'last_metrics', {}))
            metrics['snapshot_path'] = path
            metrics['snapshot_size_bytes'] = snapshot_size
            metrics['total_seconds_including_load'] = time.perf_counter() - started
            try:
                import resource
                metrics['process_peak_rss_mb'] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
            except (ImportError, AttributeError):
                metrics['process_peak_rss_mb'] = None
            metrics_path = os.path.join(
                output_dir, f'analysis_metrics_round_{round_index}.json')
            with open(metrics_path, 'w', encoding='utf-8') as handle:
                json.dump(metrics, handle, indent=2)
            print(f'[SIGN-SHADOW-OFFLINE] analyzed round={round_index} '
                  f'seconds={metrics["total_seconds_including_load"]:.3f} '
                  f'lasa_audits={metrics.get("audit_candidate_under_lasa_calls")}',
                  flush=True)
        finally:
            del snapshot
            gc.collect()
    print(f'[SIGN-SHADOW-OFFLINE] wrote results to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
