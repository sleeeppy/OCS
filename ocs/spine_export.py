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

from .config import SPINE_VERSION
from .rig import RigResult

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


def _rot(*frames: tuple[float, float]) -> list[dict]:
    """Rotation timeline from (time, degrees) pairs, eased between keys."""
    out: list[dict] = []
    for i, (t, v) in enumerate(frames):
        kf: dict = {"value": round(v, 3)}
        if t:
            kf["time"] = round(t, 4)
        if i < len(frames) - 1:
            t2, v2 = frames[i + 1]
            kf["curve"] = [round(c, 5) for c in _handles(t, v, t2, v2)]
        out.append(kf)
    return out


def _trans(*frames: tuple[float, float, float]) -> list[dict]:
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
        if i < len(frames) - 1:
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


PRESETS = {
    "idle": _idle,
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
    return caps


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


def add_animations(doc: dict, rig: RigResult, names: list[str] | None = None) -> dict:
    available = {b.name for b in rig.bones}
    caps = limb_swing_caps(rig)
    for name in (names if names is not None else list(PRESETS)):
        gen = PRESETS.get(name)
        if gen is None:
            continue
        data = _clamp_rotations(gen(available), caps)
        if data.get("bones"):
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
