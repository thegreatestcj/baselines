"""Shared per-object colours (baselines/PhysON addition): object ids are 1-based, 0 = background.

The first two entries are MOSIV's original obj1 (gold) / obj2 (green); the rest extend the palette
so that scenes with up to eight objects render distinguishably. eval/convert_physon_to_mosiv.py
carries the same list (it must not import from MOSIV/)."""

OBJ_COLORS = [
    [1.0, 0.784, 0.157],   # 1 gold
    [0.004, 0.267, 0.129], # 2 dark green
    [0.122, 0.467, 0.706], # 3 blue
    [0.839, 0.153, 0.157], # 4 red
    [0.580, 0.404, 0.741], # 5 purple
    [0.549, 0.337, 0.294], # 6 brown
    [0.890, 0.467, 0.761], # 7 pink
    [0.090, 0.745, 0.812], # 8 cyan
]
BG_COLOR = [0.5, 0.5, 0.5]


def obj_color(k):
    """float RGB in [0,1] of object id k (1-based); background (0) is grey."""
    k = int(k)
    if k <= 0:
        return list(BG_COLOR)
    return list(OBJ_COLORS[(k - 1) % len(OBJ_COLORS)])


def obj_color_u8(k):
    return [int(round(c * 255)) for c in obj_color(k)]
