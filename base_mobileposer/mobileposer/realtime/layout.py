"""Map live SlimeVR tracker bindings to a mobileposer no-head-5imu layout.

SlimeVR-Server assigns each tracker a ``BodyPart`` (see
``SlimeVR-Server/server/core/.../tracking/trackers/TrackerPosition.kt`` on the
Kotlin side, or ``solarxr_protocol.datatypes.BodyPart`` on the wire). mobileposer's
``no_head_5imu_surface`` checkpoints are trained per fixed 5-sensor layout, named
and defined in ``mobileposer.no_head_layouts.LAYOUTS``. This module bridges the two:
given the set of currently-online tracker body parts, it figures out (without any
hardcoded/static config) which one of the six trained layouts -- if any -- matches
exactly.

Only an *exact* match is accepted. These checkpoints are independently trained
per fixed combo (not a single arbitrarily-zero-maskable model), so silently
falling back to "closest" layout would produce a plausible-looking but wrong
pose -- see plan risk R1.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, List, Optional

from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.realtime.solarxr_client import TrackerSample
from solarxr_protocol.datatypes.BodyPart import BodyPart

# BodyPart -> mobileposer surface-attachment label. Only the 11 body parts used
# by the six no_head_5imu_surface layouts are mapped; everything else (hands,
# fingers, head, chest, ...) is irrelevant to this checkpoint family.
BODY_PART_TO_LABEL: Dict[int, str] = {
    BodyPart.LEFT_LOWER_ARM: "lw",    # forearm -- "wrist" sensor, see surface_imu.py
    BodyPart.RIGHT_LOWER_ARM: "rw",
    BodyPart.LEFT_UPPER_ARM: "lu",
    BodyPart.RIGHT_UPPER_ARM: "ru",
    BodyPart.LEFT_UPPER_LEG: "lt",    # thigh
    BodyPart.RIGHT_UPPER_LEG: "rt",
    BodyPart.LEFT_LOWER_LEG: "ls",    # shank
    BodyPart.RIGHT_LOWER_LEG: "rs",
    BodyPart.LEFT_FOOT: "lf",
    BodyPart.RIGHT_FOOT: "rf",
    BodyPart.WAIST: "waist",
}
LABEL_TO_BODY_PART: Dict[str, int] = {v: k for k, v in BODY_PART_TO_LABEL.items()}

# name -> frozenset(labels), derived once from the authoritative training-time
# definitions in no_head_layouts.py so this can never drift out of sync with it.
_LAYOUT_LABEL_SETS: Dict[str, FrozenSet[str]] = {
    name: frozenset(spec["labels"]) for name, spec in LAYOUTS.items()
}


class LayoutMismatchError(RuntimeError):
    """Raised when the live tracker set does not exactly match a trained layout."""


@dataclasses.dataclass(frozen=True)
class ResolvedLayout:
    name: str
    labels: List[str]                      # ordered, defines the model's input slot order
    label_to_tracker_key: Dict[str, str]    # label -> SolarXR "device:trackerNum" key


def online_labels(samples: Dict[str, TrackerSample]) -> Dict[str, str]:
    """{mobileposer label: tracker_key} for every currently-online, mappable tracker.

    Raises LayoutMismatchError if two online trackers claim the same body part
    (ambiguous binding -- this must be fixed in SlimeVR-Server, not guessed here).
    """
    result: Dict[str, str] = {}
    for sample in samples.values():
        if not sample.online:
            continue
        label = BODY_PART_TO_LABEL.get(sample.body_part)
        if label is None:
            continue  # tracker bound to a body part this checkpoint family doesn't use
        if label in result and result[label] != sample.tracker_key:
            raise LayoutMismatchError(
                f"Two online trackers are both bound to body part '{label}': "
                f"{result[label]} and {sample.tracker_key}. Fix the binding in SlimeVR-Server."
            )
        result[label] = sample.tracker_key
    return result


def detect_layout(samples: Dict[str, TrackerSample]) -> ResolvedLayout:
    """Resolve the current online tracker set to exactly one trained layout.

    Raises LayoutMismatchError if the online set matches zero or more than one
    of the six no_head_5imu_surface layouts.
    """
    labels = online_labels(samples)
    current = frozenset(labels)

    matches = [name for name, label_set in _LAYOUT_LABEL_SETS.items() if label_set == current]

    if not matches:
        known = ", ".join(f"{name}={sorted(s)}" for name, s in _LAYOUT_LABEL_SETS.items())
        raise LayoutMismatchError(
            f"Online tracker body parts {sorted(current)} do not exactly match any "
            f"trained no_head_5imu_surface layout. Known layouts: {known}"
        )
    if len(matches) > 1:
        # Cannot happen with the current six layouts (all label sets are distinct),
        # but fail loudly rather than silently picking one if that ever changes.
        raise LayoutMismatchError(f"Ambiguous layout match for {sorted(current)}: {matches}")

    name = matches[0]
    ordered_labels = LAYOUTS[name]["labels"]
    return ResolvedLayout(
        name=name,
        labels=ordered_labels,
        label_to_tracker_key={label: labels[label] for label in ordered_labels},
    )
