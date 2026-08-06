"""Write Spine 4.2 JSON from a ``RigResult``, plus a set of preset animations.

Written from the public Spine JSON format rather than adapted from any existing
implementation (see NOTICE.md). The parts worth knowing:

- ``bones[].x/y/rotation`` are **parent-local**. ``ocs.rig`` already produced
  them that way.
- A weighted mesh's ``vertices`` is a flat, variable-length stream:
  ``[boneCount, boneIndex, x, y, weight] * boneCount`` per vertex, where ``x/y``
  are in that bone's setup-local space. An unweighted mesh would instead be a
  flat ``[x, y, ...]``; OCS always weights.
- ``hull`` is a *count*, not a list: the first ``hull`` entries of ``uvs`` must be
  the outline, in order. ``ocs.rig`` emits hull points first for this reason.
- ``slots`` order **is** the draw order. Spine's separate ``drawOrder`` timeline
  only expresses deviations from it, so ordering the slot list correctly is what
  actually matters.

Animation signs: all bones are aimed down their own length, so a positive
rotation swings a limb the same way relative to that limb regardless of side.
Mirrored limbs therefore need mirrored signs, which ``_mirror`` applies.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import taxonomy
from .config import SPINE_VERSION
from .rig import Attachment, RigResult, px_to_spine
from .psd_io import Part

#: How far along the segment the bezier handles sit. 0.25/0.75 is the smooth
#: ease-in-out the Spine editor emits.
_EASE = 0.25

#: Value channels per bone timeline, which fixes how long each ``curve`` must be.
#: ``readCurve`` indexes it as ``i = value << 2``, so a two-channel timeline needs
#: 8 numbers: x's handles then y's.
_TIMELINE_CHANNELS = {
    "rotate": 1, "translatex": 1, "translatey": 1, "scalex": 1, "scaley": 1,
    "shearx": 1, "sheary": 1,
    "translate": 2, "scale": 2, "shear": 2,
}


def _handles(t1: float, v1: float, t2: float, v2: float) -> list[float]:
    """Bezier handles for one value channel, in **absolute** (time, value) space.

    Not normalised 0..1. ``readCurve`` uses ``curve[i]`` directly as a time and
    ``curve[i+1]`` as a value, so normalised numbers describe a curve through
    unrelated coordinates.
    """
    dt = t2 - t1
    return [t1 + dt * _EASE, v1, t1 + dt * (1.0 - _EASE), v2]


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob, usedforsecurity=False).hexdigest()[:11]


# --------------------------------------------------------------------------
# skeleton document
# --------------------------------------------------------------------------


def build_skeleton(rig: RigResult, name: str = "character", images: str = "./images/") -> dict:
    cw, ch = rig.canvas
    ox, oy = rig.origin_px

    bones = []
    for b in rig.bones:
        entry: dict = {"name": b.name}
        if b.parent:
            entry["parent"] = b.parent
        if abs(b.local_x) > 1e-6:
            entry["x"] = round(b.local_x, 3)
        if abs(b.local_y) > 1e-6:
            entry["y"] = round(b.local_y, 3)
        if abs(b.local_rot) > 1e-6:
            entry["rotation"] = round(b.local_rot, 3)
        if b.length > 1e-6:
            entry["length"] = round(b.length, 3)
        bones.append(entry)

    # Slot order is draw order: rig.slots is already far-to-near.
    slots = [
        {"name": s.name, "bone": s.bone, "attachment": s.attachment}
        for s in rig.slots
    ]

    attachments: dict[str, dict] = {}
    for slot in rig.slots:
        att = rig.attachments[slot.attachment]
        if att.kind == "mesh":
            body = {
                "type": "mesh",
                "uvs": [round(v, 5) for v in att.uvs],
                "triangles": att.triangles,
                "vertices": [round(v, 3) for v in att.vertices],
                "hull": att.hull,
                "width": round(att.width, 1),
                "height": round(att.height, 1),
            }
            if att.edges:
                body["edges"] = att.edges
        else:
            body = {
                "x": round(att.region_x, 3),
                "y": round(att.region_y, 3),
                "width": round(att.width, 1),
                "height": round(att.height, 1),
            }
            if abs(att.region_rotation) > 1e-6:
                body["rotation"] = round(att.region_rotation, 3)
        attachments[slot.name] = {att.name: body}

    doc = {
        "skeleton": {
            "spine": f"{SPINE_VERSION}.0",
            "x": round(-ox, 2),
            "y": round(oy - ch, 2),
            "width": cw,
            "height": ch,
            "images": images,
            "audio": "",
        },
        "bones": bones,
        "slots": slots,
        "skins": [{"name": "default", "attachments": attachments}],
        "animations": {},
    }
    doc["skeleton"]["hash"] = _hash(
        {"b": bones, "s": slots, "n": name, "a": sorted(attachments)}
    )
    return doc


# --------------------------------------------------------------------------
# animations
# --------------------------------------------------------------------------


def _rot(*frames: tuple[float, float], linear: bool = False) -> list[dict]:
    """Rotation timeline from (time, degrees) pairs.

    ``linear`` omits the curve, which is Spine's default and what a *sampled*
    curve needs. ``_handles`` eases in and out between every consecutive pair --
    right for two or three hand-placed keys, and wrong for a sine cut into a
    dozen samples, because it brings the value to a stop at each one. Twelve
    little stops a cycle is exactly what "the animation stutters" looks like.
    Straight lines between closely spaced samples read as one smooth curve.
    """
    out: list[dict] = []
    for i, (t, v) in enumerate(frames):
        kf: dict = {"value": round(v, 3)}
        if t:
            kf["time"] = round(t, 4)
        if not linear and i < len(frames) - 1:
            t2, v2 = frames[i + 1]
            kf["curve"] = [round(c, 5) for c in _handles(t, v, t2, v2)]
        out.append(kf)
    return out


def _trans(*frames: tuple[float, float, float], linear: bool = False) -> list[dict]:
    """Translation timeline from (time, x, y) triples.

    The ``curve`` here carries **8** numbers, not 4. A translate timeline has two
    value channels and ``readCurve`` is called once per channel with
    ``i = value << 2``, so x reads ``curve[0:4]`` and y reads ``curve[4:8]``.
    Emitting only 4 leaves y reading ``undefined``, and ``undefined * scale`` is
    NaN -- which lands in the bezier table, makes the bone's translation NaN, and
    propagates through every descendant's world transform. The visible symptom is
    the player refusing to start with "Animation bounds are invalid".
    """
    out: list[dict] = []
    for i, (t, x, y) in enumerate(frames):
        kf: dict = {}
        if t:
            kf["time"] = round(t, 4)
        if abs(x) > 1e-6:
            kf["x"] = round(x, 3)
        if abs(y) > 1e-6:
            kf["y"] = round(y, 3)
        if not linear and i < len(frames) - 1:
            t2, x2, y2 = frames[i + 1]
            kf["curve"] = [
                round(c, 5)
                for c in _handles(t, x, t2, x2) + _handles(t, y, t2, y2)
            ]
        out.append(kf)
    return out


def _sym(side: str, degrees: float) -> float:
    """Opposite signs per side, for motion that should look *symmetric*.

    Only correct for symmetric intent (breathing out, both knees bending). It is
    wrong for alternating motion, where the phase already encodes the opposition,
    and wrong for a single-limb gesture, which needs an absolute direction.

    Naming the two cases apart matters because the obvious blanket rule -- "left
    and right are mirrored, so flip the sign" -- is false here. A bone's frame is
    its own aim direction, and on a standing figure both legs and both arms point
    *down*, not outwards, so they share a frame rather than mirroring it. Applying
    a flip on top of an alternating phase cancelled it exactly, and both legs
    swung forward together.
    """
    return degrees if side == "left" else -degrees


def _idle(available: set[str]) -> dict:
    """Breathing sway, ~1.8 s loop. The one animation every rig should have."""
    bones: dict[str, dict] = {}
    if "torso" in available:
        bones["torso"] = {"translate": _trans((0, 0, 0), (0.9, 0, 2.5), (1.8, 0, 0))}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot((0, 0), (0.9, 1.4), (1.8, 0))}
    if "head" in available:
        bones["head"] = {"rotate": _rot((0, 0), (0.6, -1.2), (1.2, 1.2), (1.8, 0))}
    for side in ("left", "right"):
        arm, elbow = f"{side}Arm", f"{side}Elbow"
        if arm in available:
            bones[arm] = {"rotate": _rot((0, 0), (0.9, _sym(side, 2.2)), (1.8, 0))}
        if elbow in available:
            bones[elbow] = {"rotate": _rot((0, 0), (0.9, _sym(side, 1.6)), (1.8, 0))}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot((0, 0), (0.9, 2.0), (1.8, 0))}
    if "tail" in available:
        bones["tail"] = {"rotate": _rot((0, 0), (0.45, 6), (1.35, -6), (1.8, 0))}
    return {"bones": bones}


def _walk(available: set[str]) -> dict:
    """Opposing limbs plus a hip bob, 0.8 s loop."""
    bones: dict[str, dict] = {}
    if "torso" in available:
        bones["torso"] = {
            "translate": _trans((0, 0, 0), (0.2, 0, 4), (0.4, 0, 0), (0.6, 0, 4), (0.8, 0, 0))
        }
    # The phase *is* the opposition: at t=0 the left leg leads and the right
    # trails. No per-side sign flip on top of it -- that would cancel it out and
    # swing both legs the same way, which is a hop, not a walk.
    phase = {"left": 1.0, "right": -1.0}
    for side, sign in phase.items():
        leg, knee = f"{side}Leg", f"{side}Knee"
        arm, elbow = f"{side}Arm", f"{side}Elbow"
        if leg in available:
            bones[leg] = {"rotate": _rot(
                (0, 16 * sign), (0.4, -16 * sign), (0.8, 16 * sign),
            )}
        if knee in available:
            bones[knee] = {"rotate": _rot(
                (0, -6 * sign), (0.2, -22 * sign), (0.4, -4 * sign), (0.8, -6 * sign),
            )}
        # Arms counter the legs, so the opposite arm swings with each step.
        if arm in available:
            bones[arm] = {"rotate": _rot(
                (0, -12 * sign), (0.4, 12 * sign), (0.8, -12 * sign),
            )}
        if elbow in available:
            bones[elbow] = {"rotate": _rot(
                (0, 8 * sign), (0.4, -4 * sign), (0.8, 8 * sign),
            )}
    return {"bones": bones}


def _wave(available: set[str]) -> dict:
    """Right arm greeting, 1.2 s. Uses the character's right, i.e. viewer left.

    A single-limb gesture needs an absolute direction, so no per-side flip: the
    arm hangs down, and a negative rotation lifts it up and away from the body.
    """
    bones: dict[str, dict] = {}
    arm, elbow = "rightArm", "rightElbow"
    if arm in available:
        bones[arm] = {"rotate": _rot((0, 0), (0.3, -110), (1.0, -110), (1.2, 0))}
    if elbow in available:
        bones[elbow] = {"rotate": _rot(
            (0, 0), (0.3, -20), (0.5, 22), (0.7, -20), (0.9, 22), (1.2, 0),
        )}
    if "head" in available:
        bones["head"] = {"rotate": _rot((0, 0), (0.3, -5), (1.0, -5), (1.2, 0))}
    if "torso" in available:
        bones["torso"] = {"rotate": _rot((0, 0), (0.3, -2), (1.0, -2), (1.2, 0))}
    return {"bones": bones}


def _jump(available: set[str]) -> dict:
    """Squat, launch, land. 1.0 s."""
    bones: dict[str, dict] = {}
    if "torso" in available:
        bones["torso"] = {"translate": _trans(
            (0, 0, 0), (0.18, 0, -26), (0.42, 0, 96), (0.72, 0, -18), (1.0, 0, 0)
        )}
    # A squat is symmetric, so both sides mirror.
    for side in ("left", "right"):
        leg, knee = f"{side}Leg", f"{side}Knee"
        if leg in available:
            bones[leg] = {"rotate": _rot(
                (0, 0), (0.18, _sym(side, 24)), (0.42, _sym(side, -10)),
                (0.72, _sym(side, 18)), (1.0, 0),
            )}
        if knee in available:
            bones[knee] = {"rotate": _rot(
                (0, 0), (0.18, _sym(side, -40)), (0.42, _sym(side, 6)),
                (0.72, _sym(side, -30)), (1.0, 0),
            )}
        arm = f"{side}Arm"
        if arm in available:
            bones[arm] = {"rotate": _rot(
                (0, 0), (0.18, _sym(side, 20)), (0.42, _sym(side, -70)), (1.0, 0)
            )}
    return {"bones": bones}


def _turn_head(available: set[str]) -> dict:
    """Look left, look right, centre. Cheap way to see the head rig working."""
    bones: dict[str, dict] = {}
    if "head" in available:
        bones["head"] = {"rotate": _rot((0, 0), (0.5, 9), (1.2, -9), (1.8, 0))}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot((0, 0), (0.5, 4), (1.2, -4), (1.8, 0))}
    if "eyes" in available:
        bones["eyes"] = {"translate": _trans((0, 0, 0), (0.5, 3, 0), (1.2, -3, 0), (1.8, 0, 0))}
    return {"bones": bones}



# ── idle variants ────────────────────────────────────────────────────
#
# Five takes on the same problem: make a figure look alive while doing nothing.
# What separates a believable idle from a rigged one is not amplitude, it is
# structure, and all five are built from the same four rules.
#
# **Lag down the chain.** A breath starts at the trunk and reaches the neck,
# then the head, then the ends of the hair, each a little later. Everything
# arriving on the same keyframe is the single clearest tell that a machine wrote
# the animation.
#
# **No exact mirror.** One side always moves a little less than the other.
#
# **Asymmetric timing.** Drawing breath is quicker than settling out of it. A
# plain sine is symmetric in time and reads as a metronome; ``skew`` leans it.
#
# **Nothing that is resting may slide.** A hand on a skirt or under a chin has
# its contact painted into the layer beneath -- the shadow it casts, the fabric
# creased under it -- and that layer travels with a different bone. Move the hand
# a few pixels and it leaves its own shadow behind. ``_planted_tips`` finds those
# limbs by measurement and ``limb_swing_caps`` holds them to a pixel of travel,
# so none of these five animate the arms directly.


def _osc(
    amplitude: float, loop: float, *, lag: float = 0.0, cycles: int = 1,
    skew: float = 0.0, samples: int = 16, bias: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """(time, value) samples of a phase-shifted, time-skewed sine over one loop.

    ``cycles`` is a whole number so the first and last sample agree and the loop
    has no step in it. ``lag`` shifts the phase as a fraction of the loop, which
    is how the chain delay is expressed. ``skew`` bends the phase so one half of
    the cycle takes longer than the other.
    """
    frames: list[tuple[float, float]] = []
    for i in range(samples + 1):
        u = i / samples
        phase = (u * cycles + lag) % 1.0
        if skew:
            phase = phase + skew * math.sin(2.0 * math.pi * phase) / (2.0 * math.pi)
        frames.append((round(u * loop, 4),
                       amplitude * math.sin(2.0 * math.pi * phase) + bias))
    frames[-1] = (frames[-1][0], frames[0][1])
    return tuple(frames)


def _breathe_arms(
    bones: dict[str, dict], available: set[str], loop: float, amplitude: float,
    *, lag: float = 0.08, cycles: int = 1, samples: int = 16,
) -> None:
    """Give the arms a share of the breath, lagged and unmirrored.

    Arms were left out entirely while the resting hands were pinned to a pixel,
    and a body whose chest rises while both arms stay welded reads as a
    photograph with one moving part. ``limb_swing_caps`` scales whatever is asked
    for down to what the pose allows, so a resting arm still ends up small -- the
    point is that it is no longer zero.

    The elbow trails the shoulder and takes about two thirds of it, which is what
    a limb hanging off a moving trunk does, and the two sides differ so the figure
    is not symmetric about its own spine.
    """
    bias = {"left": 1.0, "right": 0.78}
    for side in ("left", "right"):
        arm, elbow = f"{side}Arm", f"{side}Elbow"
        if arm in available:
            bones[arm] = {"rotate": _rot(linear=True, *_osc(
                _sym(side, amplitude) * bias[side], loop,
                lag=lag, cycles=cycles, skew=0.25, samples=samples))}
        if elbow in available:
            bones[elbow] = {"rotate": _rot(linear=True, *_osc(
                _sym(side, amplitude * 0.66) * bias[side], loop,
                lag=lag + 0.05, cycles=cycles, skew=0.25, samples=samples))}


def _idle_breath(available: set[str]) -> dict:
    """Quiet breathing, 4.2 s. The one to use when nothing else is happening.

    Slow on purpose -- a person at rest breathes about fourteen times a minute,
    and most generated idles run at twice that. The lag is the whole design: the
    chest leads, the neck is a twentieth of a cycle behind it, the head twice
    that, the hair nearly a fifth of a cycle late and swinging wider than what
    drives it, because nothing is pulling it back.
    """
    loop = 4.2
    bones: dict[str, dict] = {}
    if "torso" in available:
        rise = _osc(3.4, loop, skew=0.38)
        bones["torso"] = {"translate": _trans(*[(t, 0.0, v + 3.4) for t, v in rise], linear=True)}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot(linear=True, *_osc(1.9, loop, lag=0.05, skew=0.32))}
    if "head" in available:
        bones["head"] = {"rotate": _rot(linear=True, *_osc(1.5, loop, lag=0.10, skew=0.32))}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot(linear=True, *_osc(4.2, loop, lag=0.19))}
    _breathe_arms(bones, available, loop, 3.4, lag=0.08)
    return {"bones": bones}


def _idle_settle(available: set[str]) -> dict:
    """Weight settling into the pose, 7.5 s. The stillest of the five.

    Two rates at once: the breath, and under it a much slower drift as the body
    finds its balance. Because 7.5 s carries two breath cycles and one drift
    cycle, the pose never repeats within the loop -- the figure returns to a
    slightly different place each breath, which is what a person holding still
    actually does. Nothing here is fast enough to read as a gesture.
    """
    loop = 7.5
    bones: dict[str, dict] = {}
    if "torso" in available:
        breath = _osc(2.8, loop, cycles=2, skew=0.4, samples=16)
        drift = _osc(2.0, loop, lag=0.28, samples=16)
        bones["torso"] = {"translate": _trans(
            *[(t, drift[i][1] * 0.8, v + 2.8) for i, (t, v) in enumerate(breath)],
            linear=True)}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot(linear=True, *_osc(1.3, loop, lag=0.06, cycles=2,
                                              skew=0.35, samples=16))}
    if "head" in available:
        fast = _osc(1.0, loop, lag=0.12, cycles=2, skew=0.35, samples=16)
        slow = _osc(1.7, loop, lag=0.33, samples=16)
        bones["head"] = {"rotate": _rot(
            *[(t, v + slow[i][1]) for i, (t, v) in enumerate(fast)], linear=True)}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot(linear=True, *_osc(3.1, loop, lag=0.22, samples=16))}
    _breathe_arms(bones, available, loop, 2.6, lag=0.09, cycles=2, samples=16)
    return {"bones": bones}


def _idle_glance(available: set[str]) -> dict:
    """Breathing with the attention wandering, 6.0 s.

    The eyes carry it. They move first and by far the most -- gaze leads a head
    turn by about a tenth of a second in life, and the head follows only part of
    the way, because you do not turn your head to look at something you are not
    interested in. Between the two glances the eyes hold still, which is what
    makes the moves read as looking rather than drifting.
    """
    loop = 6.0
    bones: dict[str, dict] = {}
    if "torso" in available:
        rise = _osc(3.0, loop, cycles=2, skew=0.36, samples=16)
        bones["torso"] = {"translate": _trans(*[(t, 0.0, v + 3.0) for t, v in rise], linear=True)}
    if "eyes" in available:
        bones["eyes"] = {"translate": _trans(
            (0, 0, 0), (0.9, 0, 0), (1.35, 2.9, 0), (2.5, 2.6, 0), (2.9, 0, 0),
            (3.6, 0, 0), (4.0, -2.4, 0), (4.9, -2.1, 0), (5.4, 0, 0), (6.0, 0, 0),
        )}
    if "head" in available:
        bones["head"] = {"rotate": _rot(
            (0, 0), (1.0, 0.6), (1.5, 4.4), (2.5, 4.0), (3.1, 0.4),
            (3.7, -0.4), (4.15, -3.4), (4.9, -3.0), (5.5, 0.2), (6.0, 0),
        )}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot(
            (0, 0), (1.1, 0.4), (1.65, 2.0), (2.6, 1.8), (3.2, 0.2),
            (3.85, -0.2), (4.3, -1.5), (5.0, -1.3), (5.6, 0), (6.0, 0),
        )}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot(
            (0, 0), (1.8, -3.3), (2.8, 1.3), (4.4, 2.9), (5.3, -1.1), (6.0, 0),
        )}
    _breathe_arms(bones, available, loop, 2.8, lag=0.1, cycles=2, samples=16)
    return {"bones": bones}


def _idle_sway(available: set[str]) -> dict:
    """A slow lateral sway with the hair as a pendulum, 5.4 s.

    The braid is the point. It hangs off the head, so it is driven by the head
    and nothing drives it back -- it arrives late, swings past, and comes around
    on its own period rather than the body's. Giving it a quarter-cycle lag and
    twice the amplitude of what moves it is what makes hair look like hair
    instead of a painted-on shape.
    """
    loop = 5.4
    bones: dict[str, dict] = {}
    if "torso" in available:
        side = _osc(3.6, loop, skew=0.2, samples=16)
        rise = _osc(2.2, loop, cycles=2, lag=0.1, skew=0.36, samples=16)
        bones["torso"] = {"translate": _trans(
            *[(t, v, rise[i][1] + 2.2) for i, (t, v) in enumerate(side)], linear=True)}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot(linear=True, *_osc(-2.2, loop, lag=0.07, samples=16))}
    if "head" in available:
        bones["head"] = {"rotate": _rot(linear=True, *_osc(-1.7, loop, lag=0.13, samples=16))}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot(linear=True, *_osc(6.8, loop, lag=0.26, samples=16))}
    _breathe_arms(bones, available, loop, 4.5, lag=0.11, samples=16)
    return {"bones": bones}


def _idle_sigh(available: set[str]) -> dict:
    """One long breath drawn and let go, 5.6 s.

    Not a loop of a single shape: the chest fills over about a second and a half
    and empties over three, and the head lifts a little on the way in and drops
    slightly below where it started on the way out before recovering. That
    difference between the two halves is the entire read -- a symmetric rise and
    fall is a machine, an asymmetric one is a person.
    """
    loop = 5.6
    bones: dict[str, dict] = {}
    if "torso" in available:
        bones["torso"] = {"translate": _trans(
            (0, 0, 0), (0.5, 0, 3.0), (1.5, 0, 7.2), (2.0, 0, 6.6),
            (3.4, 0, 1.9), (4.4, 0, -1.1), (5.1, 0, -0.4), (5.6, 0, 0),
        )}
    if "neck" in available:
        bones["neck"] = {"rotate": _rot(
            (0, 0), (0.7, 1.9), (1.7, 3.8), (2.3, 3.2),
            (3.7, 0.6), (4.6, -1.3), (5.2, -0.4), (5.6, 0),
        )}
    if "head" in available:
        bones["head"] = {"rotate": _rot(
            (0, 0), (0.9, 1.5), (1.9, 3.2), (2.5, 2.6),
            (3.9, 0.2), (4.8, -1.9), (5.3, -0.6), (5.6, 0),
        )}
    if "hairBack" in available:
        bones["hairBack"] = {"rotate": _rot(
            (0, 0), (1.2, -2.6), (2.4, 2.1), (3.2, 3.4),
            (4.3, -1.7), (5.1, 0.9), (5.6, 0),
        )}
    _breathe_arms(bones, available, loop, 4.0, lag=0.12, samples=16)
    return {"bones": bones}


#: Cloth settings per animation: (loop seconds, amplitude px, cycles, phase lag).
#: The loop must match the preset's own, or the fabric and the body drift apart.
CLOTH_MOTION = {
    "idle":        (3.6,  7.5, 1, 0.20),
    "idle_breath": (4.2,  8.5, 1, 0.22),
    "idle_settle": (7.5,  6.5, 2, 0.25),
    "idle_glance": (6.0,  7.5, 2, 0.18),
    "idle_sway":   (5.4, 15.0, 1, 0.28),
    "idle_sigh":   (5.6, 12.0, 1, 0.24),
}


PRESETS = {
    "idle": _idle,
    "idle_breath": _idle_breath,
    "idle_settle": _idle_settle,
    "idle_glance": _idle_glance,
    "idle_sway": _idle_sway,
    "idle_sigh": _idle_sigh,
    "walk": _walk,
    "wave": _wave,
    "jump": _jump,
    "turn_head": _turn_head,
}


def limb_swing_caps(rig: RigResult) -> dict[str, float]:
    """Largest rotation each limb root may take before the limb hits something.

    The presets are written as a side-on figure would move, and most illustrations
    are drawn front-on. Seen from the front, rotating a leg about the hip does not
    swing it forward -- it swings it *sideways across the body*, and the amplitude
    that looks like a stride from the side makes the legs cross.

    Measured on this character: hips 34 px apart, legs 396 px long, so the walk
    preset's +-16 deg moves each foot 109 px sideways. Two feet closing 218 px
    across a 34 px gap cross, every time. Nothing about the meshes is wrong; the
    amplitude is geometrically impossible.

    So each limb gets a cap from the rig it is actually attached to:

    - legs: the pair must not close more than the hip separation, so
      ``asin(hip_gap / (2 * leg_len))``.
    - arms: the hand must not swing past the body midline, so
      ``asin(shoulder_offset / arm_len)``.

    The cap is **directional**. Only the rotation that swings a limb *toward* the
    body collides; lifting an arm away is free, and a wave that raises it 110 deg
    is exactly what a wave should do. Capping both signs would flatten every
    gesture into a twitch. Which sign is inward is measured from the bone's own
    aim direction rather than assumed, because bones point down their own length
    and the two sides do not mirror (see ``_sym``).

    Returns ``{bone: (max_negative, max_positive)}`` in degrees, 180 where free.
    """
    pos = {b.name: (b.world_x, b.world_y) for b in rig.bones}
    aim = {b.name: b.world_rot for b in rig.bones}

    def dist(a: str, b: str) -> float:
        if a not in pos or b not in pos:
            return 0.0
        return math.dist(pos[a], pos[b])

    def cap(clearance: float, length: float) -> float:
        if length <= 1e-6:
            return 180.0
        ratio = max(0.0, min(1.0, clearance / length))
        return math.degrees(math.asin(ratio))

    midline = pos["torso"][0] if "torso" in pos else 0.0

    def directional(bone: str, clearance: float, length: float) -> tuple[float, float]:
        """(max_negative, max_positive) for ``bone``, limiting only the inward sign.

        A positive rotation moves the bone's tip by ``(-sin, cos) * length``, so the
        sign of ``-sin(world_rot)`` says which way its x travels. Inward is toward
        the midline; that direction gets the geometric cap and the other stays free.
        """
        limit = cap(clearance, length)
        dx_for_positive = -math.sin(math.radians(aim.get(bone, -90.0)))
        toward_midline = 1.0 if pos[bone][0] < midline else -1.0
        # Positive rotation is inward when its x-travel points at the midline.
        if dx_for_positive * toward_midline > 0:
            return 180.0, limit
        return limit, 180.0

    caps: dict[str, tuple[float, float]] = {}

    # --- legs -------------------------------------------------------------
    if "leftLeg" in pos and "rightLeg" in pos:
        hip_gap = abs(pos["leftLeg"][0] - pos["rightLeg"][0])
        for side in ("left", "right"):
            leg, knee, foot = f"{side}Leg", f"{side}Knee", f"{side}Foot"
            length = dist(leg, foot) or dist(leg, knee) * 2.0
            # Both legs move, so each may only spend half the gap.
            caps[leg] = directional(leg, hip_gap / 2.0, length)
            if knee in pos:
                caps[knee] = directional(
                    knee, hip_gap, dist(knee, foot) or length / 2.0)

    # --- arms -------------------------------------------------------------
    for side in ("left", "right"):
        arm, elbow, hand = f"{side}Arm", f"{side}Elbow", f"{side}Hand"
        if arm not in pos:
            continue
        clearance = abs(pos[arm][0] - midline)
        length = dist(arm, hand) or dist(arm, elbow) * 2.0
        caps[arm] = directional(arm, clearance, length)
        if elbow in pos:
            caps[elbow] = directional(
                elbow, clearance, dist(elbow, hand) or length / 2.0)

    # A resting limb is no longer pinned. Holding the hands to a pixel kept the
    # painted contact shadow perfectly aligned, but it welded the sleeves to the
    # body and the figure read as a photograph with a breathing chest. The limit
    # that matters is the seam coming apart, not the shadow drifting, and the
    # directional caps above already stop a limb swinging into the body.
    #
    # ``_planted_tips`` stays, because ``cloth_deform`` still needs to know where
    # the fabric is held: a ripple running through the hand pulls it away from its
    # own shadow far faster than the arm rotating does.
    return caps


def _planted_tips(rig: RigResult) -> list[tuple[str, str]]:
    """Limb bones whose tip lies inside another part's opaque body.

    Returns ``[(chain_bone, tip_bone)]`` for every arm or leg found resting on
    something. Parts belonging to the limb's own chain are skipped, so an arm is
    never counted as resting on its own sleeve.
    """
    pos = {b.name: (b.world_x, b.world_y) for b in rig.bones}
    origin = rig.origin_px
    out: list[tuple[str, str]] = []
    for side in ("left", "right"):
        chains = (
            ([f"{side}Arm", f"{side}Elbow"], f"{side}Hand"),
            ([f"{side}Leg", f"{side}Knee"], f"{side}Foot"),
        )
        for bones, tip in chains:
            if tip not in pos:
                continue
            px = int(round(pos[tip][0] + origin[0]))
            py = int(round(origin[1] - pos[tip][1]))
            for slot in rig.slots:
                if slot.bone in bones or slot.bone == tip:
                    continue
                part = rig.part_images.get(slot.name)
                if part is None:
                    continue
                x1, y1, x2, y2 = part.bbox
                if not (x1 <= px < x2 and y1 <= py < y2):
                    continue
                if int(part.rgba[py - y1, px - x1, 3]) < 250:
                    continue
                out.extend((b, tip) for b in bones if b in pos)
                break
    return out


def _clamp_rotations(data: dict, caps: dict[str, float]) -> dict:
    """Scale each rotate timeline down to its bone's cap, keeping its shape.

    Scaled rather than clipped: clipping flattens the peaks into a hold, which
    reads as a stutter. Scaling keeps the motion's timing and just makes it
    smaller.
    """
    for bone, timelines in (data.get("bones") or {}).items():
        bounds = caps.get(bone)
        keys = timelines.get("rotate")
        if bounds is None or not keys:
            continue
        neg_limit, pos_limit = bounds
        worst = 1.0
        for k in keys:
            v = k.get("value", 0.0)
            limit = pos_limit if v > 0 else neg_limit
            if abs(v) > limit > 1e-6:
                worst = min(worst, limit / abs(v))
        if worst >= 1.0:
            continue
        for k in keys:
            if "value" in k:
                k["value"] = round(k["value"] * worst, 3)
    return data



# ── cloth ────────────────────────────────────────────────────────────

#: Tags whose meshes get a cloth ripple. Skin and rigid features do not drape.
CLOTH_TAGS = frozenset({
    "topwear", "bottomwear", "legwear", "handwear", "back hair", "front hair",
    "tail", "wings",
})


def _cloth_field(rig: RigResult) -> tuple[float, float]:
    """(hang_y, span) -- where cloth is pinned, and how far it falls.

    Deliberately a property of the *skeleton*, not of each part. Two panels of one
    garment are separate meshes with separate bones; if each scaled its ripple by
    distance from its own bone they would disagree at the boundary they share and
    the seam would open the moment the cloth moved. Reading the profile off world
    position instead means any two meshes agree wherever they touch.
    """
    tops = [b.world_y for b in rig.bones if b.name in ("torso", "neck", "head")]
    hang = max(tops) if tops else max((b.world_y for b in rig.bones), default=0.0)
    low = min((b.world_y for b in rig.bones), default=hang - 1.0)
    return hang, max(1.0, hang - low)


#: Where the damping around a resting hand or foot is total, and where it ends.
#:
#: Dead flat out to the first radius, easing back to full motion by the second.
#: A single smooth falloff from the contact point is not enough: the interior
#: mesh vertices are ~28 px apart, so a narrow well simply falls between them and
#: the fabric ripples through the hand anyway. Measured in a 90 px box centred on
#: the resting hand, share of it moving: 34% at a 110 px smooth falloff.
#:
#: And the radius has to cover the hand, not the hand *bone*. A bone is a point;
#: the thing resting on the fabric is a whole palm and four fingers, and on this
#: character the fingertips reach 196-276 px from the bone. At a 95 px dead zone
#: they sat in open fabric taking the full ripple, which is what squashed them.
_CONTACT_DEAD_PX = 210.0
_CONTACT_HOLD_PX = 380.0


def _cloth_profile(
    att: Attachment, part: Part, field: tuple[float, float],
    origin: tuple[float, float], held: list[tuple[float, float]],
) -> np.ndarray:
    """Per-vertex freedom, 0 where the cloth is held and 1 where it hangs free.

    Cloth is pinned at the top and loosest at the bottom, so the amplitude has to
    grow as it falls -- a skirt that moves at the waist as much as at the hem
    looks like a flag, not a skirt.

    And it is pinned again wherever something rests on it. Without that the ripple
    runs straight through the hand lying on the skirt, which is worse than not
    moving at all: the shadow the hand casts is painted into the layer underneath
    and does not ripple with it, so the hand and its own shadow come apart. The
    first pass moved the region around the resting hand by up to 2367 px.
    """
    hang, span = field
    uv = np.asarray(att.uvs, dtype=np.float64).reshape(-1, 2)
    ox, oy = part.offset
    px = uv * np.array([att.width, att.height]) + np.array([ox, oy])
    world = np.array([px_to_spine((float(x), float(y)), origin) for x, y in px])
    freedom = np.clip((hang - world[:, 1]) / span, 0.0, 1.0)

    for hx, hy in held:
        d = np.linalg.norm(world - np.array([hx, hy])[None, :], axis=1)
        t = np.clip((d - _CONTACT_DEAD_PX)
                    / max(1.0, _CONTACT_HOLD_PX - _CONTACT_DEAD_PX), 0.0, 1.0)
        # Smoothstep, so the fabric eases back into motion instead of creasing
        # along a line where the damping stops.
        freedom *= t * t * (3.0 - 2.0 * t)
    return freedom


def cloth_deform(
    rig: RigResult, loop: float, amplitude: float, *,
    cycles: int = 1, samples: int = 16, lag: float = 0.0, travel: float = 0.55,
) -> dict:
    """Deform timelines that let the garments move without moving a bone.

    This is the answer to "the arms and the clothes do not move". They could not
    move by rotation: both of this character's hands are resting on something, and
    a hand that slides even a few pixels leaves the shadow painted under it
    behind. But a mesh does not need its bone to turn -- Spine can push the
    vertices directly, which is what a deform timeline is, and a sleeve can swing
    while the hand inside it stays exactly where it is.

    Three things make the result look like cloth rather than a wobbling sheet:

    - **Pinned at the top.** The offset scales with distance from the point the
      panel hangs from, squared, so the shoulder is still and the hem moves most.
    - **The ripple travels.** Phase is delayed by that same distance, so a wave
      runs down the fabric instead of the whole panel pulsing together.
    - **Mostly sideways.** Cloth swings across gravity, so the vertical component
      is a third of the horizontal.

    The format is per *bone influence*, not per vertex: a weighted mesh stores
    ``[x, y, weight]`` per influence and the runtime adds one offset pair to each
    as it accumulates, so a vertex with three bones needs its offset three times.
    Getting that wrong silently shifts every vertex after the first weighted one.
    """
    out: dict[str, dict] = {}
    field = _cloth_field(rig)
    origin = rig.origin_px
    pos = {b.name: (b.world_x, b.world_y) for b in rig.bones}
    held = [pos[tip] for _bone, tip in _planted_tips(rig) if tip in pos]

    for slot in rig.slots:
        att = rig.attachments.get(slot.attachment)
        part = rig.part_images.get(slot.name)
        if att is None or part is None or att.kind != "mesh" or not att.vertices:
            continue
        if taxonomy.base_tag(part.name) not in CLOTH_TAGS:
            continue
        freedom = _cloth_profile(att, part, field, origin, held)
        if not freedom.any():
            continue

        # How many influences each vertex has, so the offsets can be repeated.
        counts: list[int] = []
        i = 0
        while i < len(att.vertices):
            n = int(att.vertices[i])
            counts.append(n)
            i += 1 + n * 4
        if len(counts) != freedom.shape[0]:
            continue

        frames: list[dict] = []
        for k in range(samples + 1):
            u = k / samples
            t = u * loop
            values: list[float] = []
            for vi, n in enumerate(counts):
                theta = 2.0 * math.pi * (
                    u * cycles + lag - travel * float(freedom[vi]))
                swing = amplitude * (float(freedom[vi]) ** 2) * math.sin(theta)
                dx, dy = round(swing, 3), round(swing * 0.34, 3)
                values.extend([dx, dy] * n)
            frame: dict = {"offset": 0, "vertices": values}
            if t:
                frame["time"] = round(t, 4)
            frames.append(frame)
        # Close the loop exactly.
        frames[-1]["vertices"] = list(frames[0]["vertices"])
        if any(any(v) for v in (f["vertices"] for f in frames)):
            out.setdefault(slot.name, {})[slot.attachment] = {"deform": frames}

    return {"default": out} if out else {}


def add_animations(doc: dict, rig: RigResult, names: list[str] | None = None) -> dict:
    available = {b.name for b in rig.bones}
    caps = limb_swing_caps(rig)
    for name in (names if names is not None else list(PRESETS)):
        gen = PRESETS.get(name)
        if gen is None:
            continue
        data = _clamp_rotations(gen(available), caps)
        if name in CLOTH_MOTION:
            loop, amp, cycles, lag = CLOTH_MOTION[name]
            deform = cloth_deform(rig, loop, amp, cycles=cycles, lag=lag)
            if deform:
                # Spine 4.x nests these under "attachments", by skin then slot then
                # attachment, with "deform" as the timeline key inside. The 3.8
                # layout -- a top-level "deform" -- is accepted silently and
                # produces an animation with no deform timelines at all.
                data["attachments"] = deform
        if data.get("bones") or data.get("attachments"):
            doc["animations"][name] = data
    return doc


# --------------------------------------------------------------------------


def export_skeleton(
    rig: RigResult,
    out_path: str | Path,
    name: str = "character",
    animations: list[str] | None = None,
) -> Path:
    doc = build_skeleton(rig, name=name)
    add_animations(doc, rig, animations)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path


def validate(doc: dict) -> list[str]:
    """Structural checks. Cheap, and catches the mistakes that break the player."""
    problems: list[str] = []

    bones = doc.get("bones", [])
    names = [b["name"] for b in bones]
    if not names or names[0] != "root":
        problems.append("first bone must be 'root'")
    if len(set(names)) != len(names):
        problems.append("duplicate bone names")

    seen: set[str] = set()
    for b in bones:
        parent = b.get("parent")
        if parent is not None and parent not in seen:
            problems.append(f"bone {b['name']}: parent '{parent}' not defined earlier")
        seen.add(b["name"])

    bone_set = set(names)
    slot_names = set()
    for s in doc.get("slots", []):
        if s["bone"] not in bone_set:
            problems.append(f"slot {s['name']}: unknown bone {s['bone']}")
        if s["name"] in slot_names:
            problems.append(f"duplicate slot {s['name']}")
        slot_names.add(s["name"])

    skins = doc.get("skins", [])
    attachments = skins[0]["attachments"] if skins else {}
    for slot_name, entries in attachments.items():
        if slot_name not in slot_names:
            problems.append(f"attachment for unknown slot {slot_name}")
        for att_name, body in entries.items():
            if body.get("type") != "mesh":
                continue
            uvs, tris, verts = body["uvs"], body["triangles"], body["vertices"]
            n = len(uvs) // 2
            if body.get("hull", 0) > n:
                problems.append(f"{att_name}: hull {body['hull']} exceeds {n} vertices")
            if tris and max(tris) >= n:
                problems.append(f"{att_name}: triangle index out of range")
            if len(tris) % 3:
                problems.append(f"{att_name}: triangles not a multiple of 3")

            # Walk the weighted-vertex stream and confirm it describes exactly n
            # vertices with weights summing to 1.
            i = count = 0
            while i < len(verts):
                bone_count = int(verts[i])
                i += 1
                if bone_count < 1 or i + bone_count * 4 > len(verts):
                    problems.append(f"{att_name}: malformed vertex stream at {i}")
                    break
                total = 0.0
                for k in range(bone_count):
                    idx = int(verts[i + k * 4])
                    total += verts[i + k * 4 + 3]
                    if idx < 0 or idx >= len(names):
                        problems.append(f"{att_name}: bone index {idx} out of range")
                if abs(total - 1.0) > 0.02:
                    problems.append(f"{att_name}: vertex weights sum to {total:.3f}")
                    break
                i += bone_count * 4
                count += 1
            else:
                if count != n:
                    problems.append(f"{att_name}: {count} weighted vertices for {n} uvs")

    for anim_name, anim in doc.get("animations", {}).items():
        for bone_name, timelines in anim.get("bones", {}).items():
            if bone_name not in bone_set:
                problems.append(f"animation {anim_name}: unknown bone {bone_name}")
            for timeline_name, keys in timelines.items():
                channels = _TIMELINE_CHANNELS.get(timeline_name)
                if channels is None:
                    problems.append(
                        f"animation {anim_name}/{bone_name}: unknown timeline "
                        f"'{timeline_name}'"
                    )
                    continue
                # A curve short by one channel is the worst kind of bug: the
                # document parses, then the runtime reads past the end, gets
                # undefined, and NaN spreads through every descendant bone.
                want = channels * 4
                for i, key in enumerate(keys):
                    curve = key.get("curve")
                    if curve is None or curve == "stepped":
                        continue
                    if not isinstance(curve, list) or len(curve) != want:
                        problems.append(
                            f"animation {anim_name}/{bone_name}/{timeline_name} "
                            f"key {i}: curve has "
                            f"{len(curve) if isinstance(curve, list) else curve} "
                            f"values, needs {want} ({channels} channel(s) x 4)"
                        )
                if keys and keys[-1].get("curve") is not None:
                    problems.append(
                        f"animation {anim_name}/{bone_name}/{timeline_name}: last "
                        "key has a curve but nothing follows it"
                    )

    problems.extend(_non_finite(doc))
    return problems


def _non_finite(node, path: str = "$") -> list[str]:
    """Every non-finite float, with its path. NaN is not valid JSON."""
    import math

    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_non_finite(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_non_finite(v, f"{path}[{i}]"))
    elif isinstance(node, float) and not math.isfinite(node):
        out.append(f"non-finite value at {path}: {node}")
    return out
