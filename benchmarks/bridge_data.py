"""Common data bridge from MobilePoser processed files to legacy IMU methods.

The processed datasets use six sensor slots in a fixed order:
left wrist, right wrist, left thigh, right thigh, head, pelvis.  Legacy
methods that consume the original DIP/AMASS ``test.pt`` convention receive the
same order from :func:`export_legacy_test_file`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = PROJECT_ROOT / "base_mobileposer"
SENSOR_ORDER = ("left_wrist", "right_wrist", "left_thigh", "right_thigh", "head", "pelvis")


@dataclass
class ImuSequence:
    """One six-sensor sequence in the shared SMPL coordinate convention."""

    acc: torch.Tensor
    ori: torch.Tensor
    pose: torch.Tensor
    tran: torch.Tensor
    source: str


def _load_file(path: Path, min_frames: int, max_frames: Optional[int]) -> List[ImuSequence]:
    data = torch.load(path, map_location="cpu")
    required = ("acc", "ori", "pose", "tran")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("{} is missing {}".format(path, ", ".join(missing)))
    sequences = []
    for index, values in enumerate(zip(*(data[key] for key in required))):
        acc, ori, pose, tran = values
        length = min(len(acc), len(ori), len(pose), len(tran))
        if max_frames is not None:
            length = min(length, max_frames)
        if length < min_frames:
            continue
        if acc.ndim != 3 or acc.shape[1:] != (6, 3):
            raise ValueError("{} sequence {} has acc shape {}, expected [T, 6, 3]".format(path, index, tuple(acc.shape)))
        if ori.ndim != 4 or ori.shape[1:] != (6, 3, 3):
            raise ValueError("{} sequence {} has ori shape {}, expected [T, 6, 3, 3]".format(path, index, tuple(ori.shape)))
        if pose.ndim != 4 or pose.shape[1:] != (24, 3, 3):
            raise ValueError("{} sequence {} has pose shape {}, expected [T, 24, 3, 3]".format(path, index, tuple(pose.shape)))
        sequences.append(ImuSequence(
            acc=acc[:length].float().contiguous(),
            ori=ori[:length].float().contiguous(),
            pose=pose[:length].float().contiguous(),
            tran=tran[:length].float().contiguous(),
            source="{}#{}".format(path.name, index),
        ))
    return sequences


def load_dip_sequences(processed_root: Path, min_frames: int = 1, max_frames: Optional[int] = None) -> List[ImuSequence]:
    """Load the real DIP-IMU test split from the shared processed-data root."""
    return _load_file(processed_root / "eval" / "dip_test.pt", min_frames, max_frames)


def load_amass_sequences(
    processed_root: Path,
    min_frames: int = 1,
    max_frames: Optional[int] = None,
    max_sequences: Optional[int] = None,
) -> List[ImuSequence]:
    """Load AMASS datasets, excluding real-data eval files and ineligible files."""
    sequences: List[ImuSequence] = []
    for path in sorted(processed_root.glob("*.pt")):
        sequences.extend(_load_file(path, min_frames, max_frames))
        if max_sequences is not None and len(sequences) >= max_sequences:
            return sequences[:max_sequences]
    return sequences


def rotation_matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 3, 3]`` rotation matrices to axis-angle without SciPy."""
    diagonal = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((diagonal - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vector = torch.stack((
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ), dim=-1)
    scale = angle / (2.0 * torch.sin(angle)).clamp_min(1e-7)
    axis_angle = vector * scale.unsqueeze(-1)
    # First-order approximation avoids numerical instability around identity.
    small = angle < 1e-4
    axis_angle = torch.where(small.unsqueeze(-1), 0.5 * vector, axis_angle)
    return axis_angle


def export_legacy_test_file(sequences: Iterable[ImuSequence], output_path: Path) -> None:
    """Write the native PIP/TransPose ``test.pt`` dictionary convention."""
    items = list(sequences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Dict insertion order is intentional: older evaluators unpack ``.values()``.
    payload = {
        "acc": [item.acc for item in items],
        "ori": [item.ori for item in items],
        "pose": [rotation_matrix_to_axis_angle(item.pose).reshape(len(item.pose), -1) for item in items],
        "tran": [item.tran for item in items],
    }
    torch.save(payload, output_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export processed MobilePoser data for legacy IMU methods.")
    parser.add_argument("--processed-root", type=Path, default=BASE_ROOT / "data/processed_datasets")
    parser.add_argument("--split", choices=("dip", "amass"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args(argv)
    if args.split == "dip":
        sequences = load_dip_sequences(args.processed_root, args.min_frames, args.max_frames)
    else:
        sequences = load_amass_sequences(
            args.processed_root, args.min_frames, args.max_frames, args.max_sequences
        )
    if args.max_sequences is not None:
        sequences = sequences[:args.max_sequences]
    export_legacy_test_file(sequences, args.output)
    print("Exported {} {} sequences to {}".format(len(sequences), args.split, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
