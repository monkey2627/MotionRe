"""Read local original DIP files, bypassing all legacy processed bridges.

Mappings cross-checked against PNP/process.py process_dipimu. Local upstream
provenance is recorded separately; this module does not repair missing frames
using future samples. DIP provides no measured global root translation.
"""
import pickle
from pathlib import Path

import numpy as np

from .records import FIVE, ImuRecord, causal_fill


DIP_INDEX = {"pelvis": 2, "left_forearm": 7, "right_forearm": 8,
             "left_shank": 11, "right_shank": 12, "head": 0,
             "left_thigh": 9, "right_thigh": 10}
DIP_HZ = 60.0


def load_original(path):
    # Only use trusted local dataset files: Python pickle is executable format.
    with Path(path).open("rb") as f:
        return pickle.load(f, encoding="latin1")


def measurements(data, source, names=FIVE):
    ids = [DIP_INDEX[n] for n in names]
    r = np.asarray(data["imu_ori"][:, ids], dtype=np.float32)
    a = np.asarray(data["imu_acc"][:, ids], dtype=np.float32)
    r, a, valid = causal_fill(r, a)
    return ImuRecord(np.arange(len(r), dtype=np.float64) / DIP_HZ,
                     tuple(names), r, a, valid, str(source)).validate()


def split_for_subject(subject):
    """Reserve official test subjects; split original train by subject."""
    i = int(subject.split("_")[-1])
    if 1 <= i <= 6:
        return "train"
    if i in (7, 8):
        return "validation"
    if i in (9, 10):
        return "test_locked"
    raise ValueError("Unknown subject: " + subject)
