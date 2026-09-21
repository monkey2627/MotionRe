"""Create no-head evaluation inputs directly from an ours SMPL GT file.

This intentionally has no tracker/IMU path: acceleration and orientation come
from the same FK call as the GT, using exactly ``no_head_layouts._syn_acc`` and
the layout joint IDs.  It is therefore a reproducible rotation sanity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.fit_ours_smpl import _foot_ground_probs
from mobileposer.no_head_layouts import LAYOUTS, _syn_acc


# Same AMASS-to-runtime basis used during no-head dataset generation.
TRAINING_WORLD_ROT = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _to_training_coordinates(pose: torch.Tensor, tran: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = TRAINING_WORLD_ROT.to(pose)
    converted = pose.clone()
    converted[:, 0] = rotation @ converted[:, 0]
    return converted, (rotation @ tran.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict FK no-head sanity data from ours SMPL GT.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=["wrists_shanks_waist"])
    args = parser.parse_args()

    unknown = [layout for layout in args.layouts if layout not in LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown layouts: {unknown}")
    source = torch.load(args.source, map_location="cpu")
    body = ParametricModel(paths.smpl_file)
    metadata = source.get("metadata", source.get("fit_metadata", []))

    for layout_name in args.layouts:
        out = {"layout": layout_name, "layout_labels": LAYOUTS[layout_name]["labels"],
               "layout_joints": LAYOUTS[layout_name]["joints"],
               "pose": [], "tran": [], "shape": [], "joint": [], "contact": [],
               "acc": [], "ori": [], "metadata": []}
        joint_ids = torch.tensor(LAYOUTS[layout_name]["joints"] + [0])
        for index, (pose, tran, shape) in enumerate(zip(source["pose"], source["tran"], source["shape"])):
            pose, tran = _to_training_coordinates(pose.float(), tran.float())
            global_rotation, joint = body.forward_kinematics(pose, shape.float(), tran)
            acc = _syn_acc(joint[:, joint_ids])
            ori = global_rotation[:, joint_ids]
            # Fail fast if a future refactor breaks the claimed strict FK contract.
            if not torch.equal(acc, _syn_acc(joint[:, joint_ids])):
                raise RuntimeError("Synthetic acceleration is not reproducible")
            out["pose"].append(pose)
            out["tran"].append(tran)
            out["shape"].append(shape.float())
            out["joint"].append(joint[:, :24].contiguous())
            out["contact"].append(_foot_ground_probs(joint))
            out["acc"].append(acc)
            out["ori"].append(ori)
            source_meta = metadata[index] if index < len(metadata) else {}
            out["metadata"].append({"source": str(args.source), "source_metadata": source_meta,
                                    "signal_source": "strict_fk_sanity"})
        output_path = args.output_root / layout_name / "ours.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, output_path)
        print(f"Saved strict FK sanity input: {output_path}")


if __name__ == "__main__":
    main()
