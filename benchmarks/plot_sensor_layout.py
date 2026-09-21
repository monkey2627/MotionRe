"""Render method-specific IMU layouts on a front-view SMPL mesh."""
from pathlib import Path
import pickle
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

COMMON = [(18, "S0 left forearm"), (19, "S1 right forearm"),
          (1, "S2 left hip"), (2, "S3 right hip"),
          (15, "S4 head"), (0, "S5 pelvis")]
NATIVE_DIP = [(18, "S0 left forearm/wrist"), (19, "S1 right forearm/wrist"),
              (1, "S2 left hip/thigh"), (2, "S3 right hip/thigh"),
              (15, "S4 head"), (0, "S5 pelvis/root")]
LAYOUTS = {
    "mobileposer": ("MobilePoser (benchmark bridge)", "process.py ji_mask=[18,19,1,2,15,0]", COMMON),
    "pip": ("PIP (native DIP preprocessing)", "preprocess.py imu_mask=[7,8,11,12,0,2]", NATIVE_DIP),
    "pnp": ("PNP (native DIP preprocessing)", "process.py imu_mask=[7,8,11,12,0,2]", NATIVE_DIP),
    "transpose": ("TransPose (native DIP preprocessing)", "preprocess.py imu_mask=[7,8,11,12,0,2]", NATIVE_DIP),
    "globalpose": ("GlobalPose (native DIP preprocessing)", "process.py imu_mask=[7,8,11,12,0,2]", NATIVE_DIP),
    "imucoco": ("IMUCoCo (shared AMASS slots)", "evaluate_drift.py [L_wrist,R_wrist,L_hip,R_hip,Head,Pelvis]", COMMON),
    "slimevr": ("SliMeVR FK (shared AMASS slots)", "SENSOR_TO_JOINT=[18,19,1,2,15,0]", COMMON),
}


def load_smpl():
    path = Path(__file__).resolve().parents[1] / "PNP" / "models" / "SMPL_male.pkl"
    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")
    chumpy_ch.Ch = type("Ch", (object,), {})
    chumpy.ch = chumpy_ch
    sys.modules.setdefault("chumpy", chumpy)
    sys.modules.setdefault("chumpy.ch", chumpy_ch)
    with path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    vertices = np.asarray(model["v_template"], dtype=float)
    faces = np.asarray(model["f"], dtype=np.int64)
    joints = np.asarray(model["J_regressor"].dot(vertices), dtype=float)
    return vertices, faces, joints


def render(name, title, source, sensors, vertices, faces, joints, out_dir):
    fig, ax = plt.subplots(figsize=(6, 9), facecolor="#101318")
    # SMPL uses X horizontal and Y vertical; depth Z is intentionally collapsed.
    polygons = [vertices[face][:, [0, 1]] for face in faces]
    ax.add_collection(PolyCollection(polygons, facecolor="#79A9D6",
                                     edgecolor="#B8D0E8", linewidths=0.05, alpha=0.32))
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    ax.set_xlim(x_min - 0.22, x_max + 0.22)
    ax.set_ylim(y_min - 0.12, y_max + 0.16)
    for idx, (joint, label) in enumerate(sensors):
        x, y = joints[joint][0], joints[joint][1]
        ax.scatter([x], [y], s=125, color="#FF8A00", edgecolor="white",
                   linewidth=1.0, zorder=5)
        ax.annotate("S{}".format(idx), (x, y), xytext=(0, 0),
                    textcoords="offset points", color="black", fontsize=8,
                    fontweight="bold", ha="center", va="center", zorder=6)
    ax.set_title(title + "\n" + source, color="white", fontsize=10, pad=10)
    ax.set_aspect("equal"); ax.axis("off")
    legend_lines = ["S{} {}".format(i, label.replace("S{} ".format(i), "", 1))
                    for i, (_, label) in enumerate(sensors)]
    fig.text(0.08, 0.035, "    ".join(legend_lines[:3]), color="white", fontsize=8, ha="left")
    fig.text(0.08, 0.015, "    ".join(legend_lines[3:]), color="white", fontsize=8, ha="left")
    fig.subplots_adjust(left=0, right=1, bottom=0.08, top=0.88)
    out = out_dir / "sensor_layout_{}_smpl_front.png".format(name)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    out_dir = Path(__file__).resolve().parents[1] / "benchmark_results_analysis_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, joints = load_smpl()
    for name, (title, source, sensors) in LAYOUTS.items():
        render(name, title, source, sensors, vertices, faces, joints, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
