"""Measure reproducible MobilePoser CPU deployment characteristics.

Energy is deliberately reported as unavailable unless the host exposes Intel
RAPL.  Estimating energy from wall time would fabricate a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import torch


def _model_bytes(model: torch.nn.Module, dtype: torch.dtype) -> int:
    return sum(parameter.numel() * torch.tensor([], dtype=dtype).element_size()
               for parameter in model.parameters())


def _rapl_energy_uj() -> Optional[int]:
    paths = tuple(Path('/sys/class/powercap').glob('intel-rapl*/energy_uj'))
    if not paths:
        return None
    try:
        return sum(int(path.read_text().strip()) for path in paths)
    except OSError:
        return None


def _rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Measure MobilePoser CPU inference and model footprint.')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.runs < 1:
        parser.error('--frames and --runs must be positive')

    torch.set_num_threads(1)
    import mobileposer.config as mobile_config
    mobile_config.model_config.device = torch.device('cpu')
    from mobileposer.models import MobilePoserNet
    model = MobilePoserNet().cpu()
    checkpoint = torch.load(str(args.model), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # Lightning checkpoints may namespace parameters with ``model.``.
    if isinstance(state_dict, dict) and all(key.startswith('model.') for key in state_dict):
        state_dict = {key[6:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    input_ = torch.zeros(1, args.frames, 60, dtype=torch.float32)

    def infer():
        model.reset()
        with torch.inference_mode():
            model.forward_offline(input_, [args.frames])

    for _ in range(args.warmup):
        infer()
    rss_before = _rss_bytes()
    energy_before = _rapl_energy_uj()
    started = time.perf_counter()
    for _ in range(args.runs):
        infer()
    elapsed = time.perf_counter() - started
    energy_after = _rapl_energy_uj()
    rss_after = _rss_bytes()

    quantized_size = None
    try:
        quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(quantized.state_dict(), temporary)
        quantized_size = temporary.stat().st_size
        temporary.unlink()
    except (RuntimeError, TypeError, AttributeError):
        # Some custom recurrent modules cannot be dynamically quantised.
        quantized_size = None

    output = {
        'method': 'mobileposer', 'device': 'cpu', 'threads': 1,
        'input_frames': args.frames, 'runs': args.runs,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'fp32_model_bytes': _model_bytes(model, torch.float32),
        'fp16_model_bytes_estimate': _model_bytes(model, torch.float16),
        'int8_dynamic_state_dict_bytes': quantized_size,
        'mean_sequence_latency_ms': elapsed * 1000.0 / args.runs,
        'mean_frame_latency_ms': elapsed * 1000.0 / (args.runs * args.frames),
        'peak_rss_bytes_approx': max(value for value in (rss_before, rss_after) if value is not None)
            if rss_before is not None or rss_after is not None else None,
        'energy_uj': None if energy_before is None or energy_after is None else energy_after - energy_before,
        'energy_note': 'Measured through Intel RAPL when available; otherwise unavailable.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
