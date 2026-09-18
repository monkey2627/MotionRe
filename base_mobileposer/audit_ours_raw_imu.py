"""Audit the parallel SolarXR/raw-IMU recording in an ours capture."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    d = json.loads(args.path.read_text(encoding="utf-8"))
    frames = d.get("frames", [])
    times = torch.tensor([float(f.get("timeSeconds", i)) for i, f in enumerate(frames)])
    dt = times[1:] - times[:-1]
    print(f"file={args.path}")
    print(f"formatVersion={d.get('formatVersion')} sampleRate={d.get('sampleRate')} frames={len(frames)} duration={d.get('durationSeconds')}")
    if len(dt): print(f"dt mean={dt.mean():.6f}s median={dt.median():.6f}s min={dt.min():.6f}s max={dt.max():.6f}s")
    roles = sorted({s.get('trackerRole') for f in frames for s in f.get('sensors', []) if s.get('trackerRole')})
    print("roles:")
    for role in roles:
        samples = [next((s for s in f.get('sensors', []) if s.get('trackerRole') == role), None) for f in frames]
        online = torch.tensor([bool(s and s.get('online', False)) for s in samples])
        kinds = sorted({str(s.get('accelerationKind')) for s in samples if s})
        sources = sorted({str(s.get('source')) for s in samples if s})
        euler = torch.tensor([[float((s or {}).get('eulerAnglesDeg', {}).get(k, 0.0)) for k in ('x','y','z')] for s in samples])
        jumps = ((euler[1:] - euler[:-1] + 180) % 360 - 180).norm(dim=1) if len(euler)>1 else torch.zeros(0)
        print(f"  {role:18s} samples={sum(s is not None for s in samples):4d} online={online.float().mean():.3f} "
              f"accelerationKind={kinds} source={sources} euler_jump_p95={float(jumps.quantile(.95)) if len(jumps) else 0:.2f}deg")
    print("first sensor keys:", sorted(frames[0]['sensors'][0].keys()) if frames and frames[0].get('sensors') else [])

if __name__ == '__main__': main()
