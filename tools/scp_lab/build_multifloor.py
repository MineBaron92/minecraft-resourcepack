#!/usr/bin/env python3
"""Build the multi-floor SCP lab structures from the single-floor prefabs.

Inputs (tools/scp_lab/source/):
  scp_lab_tier1.nbt  - original surface lab (L1/L2 gates), 47x11x47
  scp_deep_lab.nbt   - original deep lab (L3-L6 gates), 47x11x47

Outputs (kubejs/data/deceasedcraft/structures/):
  scp_lab_surface.nbt          1F (start piece, surface)
  scp_lab_levels.nbt           B1 + B2 (jigsaw child, underground)
  scp_deep_lab_entrance.nbt    surface entrance hut (start piece)
  scp_deep_lab_shaft.nbt       ladder shaft (jigsaw child)
  scp_deep_lab_levels.nbt      B1 + B2 + B3 (jigsaw child, underground)

Every floor is 9 blocks tall (floor y0, room y1-7, ceiling y8). Floors are
connected by a switchback stairwell module whose top door is a keycard gate.
Pieces are joined with vertical jigsaw blocks so the surface piece keeps its
own terrain adaptation.

Requires: pip install nbtlib
Run from anywhere: python3 tools/scp_lab/build_multifloor.py
"""
import os
import sys

import nbtlib
from nbtlib import tag

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'source')
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'kubejs', 'data', 'deceasedcraft', 'structures'))

DATA_VERSION = 3465  # 1.20.1
SIZE = 47            # prefab footprint
FLOOR_H = 9          # y0 floor .. y8 ceiling

AIR = ('minecraft:air', {})
BARS_EW = ('minecraft:iron_bars', {'east': 'true', 'west': 'true', 'north': 'false', 'south': 'false', 'waterlogged': 'false'})
STAIRS = 'minecraft:polished_andesite_stairs'
TILES = ('scp_the_facility:foundation_floor_tiles', {})
CEILING = ('scp_the_facility:foundation_ceiling', {})
METAL = ('scp_the_facility:metal_block', {})
GRAY_WALL = ('scp_the_facility:foundation_gray_big_line_wall', {})
LIGHT = ('scp_the_facility:light_on', {})

# Blocks the module builders may overwrite without a warning.
STRUCTURAL = ('air', 'wall', 'metal_block', 'floor', 'ceiling', 'glass', 'iron_bars', 'reader')

DIRS = {(0, -1): 'north', (0, 1): 'south', (-1, 0): 'west', (1, 0): 'east'}


def reader_id(level, side):
    if level == 1:
        return f'scp_additions:{side}_reader'
    return f'scp_additions:lv_{level}_{side}_reader'


class Grid:
    """Sparse block grid in combined structure coordinates."""

    def __init__(self, name):
        self.name = name
        self.cells = {}  # (x, y, z) -> (block_name, props, nbt_or_None)
        self.warnings = []

    def get(self, x, y, z):
        return self.cells.get((x, y, z), (AIR[0], {}, None))

    def set(self, x, y, z, block, nbt=None, force=False):
        cur = self.cells.get((x, y, z))
        if cur and not force:
            cur_name = cur[0].split(':')[1]
            if not any(k in cur_name for k in STRUCTURAL) and cur[0] != block[0]:
                self.warnings.append(f'{self.name}: overwrote {cur[0]} at {(x, y, z)} with {block[0]}')
        self.cells[(x, y, z)] = (block[0], dict(block[1]), nbt)

    def load_floor(self, path, y_offset, transform, y_range, remap=None):
        """Copy a prefab into the grid.

        transform: 'none' | 'mirror_x' | 'mirror_z' | 'rot180'
        remap(block_name, props, nbt, orig_pos) -> (block_name, props, nbt)
        """
        src = nbtlib.load(path)
        pal = src['palette']
        mx = transform in ('mirror_x', 'rot180')
        mz = transform in ('mirror_z', 'rot180')
        for b in src['blocks']:
            x, y, z = (int(v) for v in b['pos'])
            if y not in y_range:
                continue
            p = pal[int(b['state'])]
            name = str(p['Name'])
            props = {str(k): str(v) for k, v in p.get('Properties', {}).items()}
            nbt = b.get('nbt')
            if remap:
                name, props, nbt = remap(name, props, nbt, (x, y, z))
            props = mirror_props(props, mx, mz)
            tx = SIZE - 1 - x if mx else x
            tz = SIZE - 1 - z if mz else z
            self.cells[(tx, y + y_offset, tz)] = (name, props, nbt)


def mirror_props(props, mx, mz):
    swap = {}
    if mx:
        swap.update({'east': 'west', 'west': 'east'})
    if mz:
        swap.update({'north': 'south', 'south': 'north'})
    if not swap:
        return props
    out = {}
    for k, v in props.items():
        if k == 'facing':
            out[k] = swap.get(v, v)
        elif k in swap:
            out[swap[k]] = v
        else:
            out[k] = v
    return out


def is_wall(block_name):
    n = block_name.split(':')[1]
    return 'wall' in n or n == 'metal_block' or 'glass' in n


# ---------------------------------------------------------------------------
# Stairwell module
#
# Local frame: a = across (0..3), b = along (0..7). x = ox + ax * a,
# z = oz + bz * b (the frame is axis aligned). Flight 1 descends along +b on
# a0-1, turns at the landing (b6-7) and flight 2 returns along -b on a2-3.
# The upper floor sees a 2-wide gated corridor (a0-1, b-1..6); the lower floor
# gets the full enclosure (a-1..4, b-1..8) with an open exit at b-1, a2-3.
# ---------------------------------------------------------------------------

class Module:
    def __init__(self, ox, oz, ax, bz):
        self.ox, self.oz, self.ax, self.bz = ox, oz, ax, bz

    def xz(self, a, b):
        return self.ox + self.ax * a, self.oz + self.bz * b

    def dir_of_b(self, sign):
        return DIRS[(0, self.bz * sign)]


def stairs(facing):
    return (STAIRS, {'facing': facing, 'half': 'bottom', 'shape': 'straight', 'waterlogged': 'false'})


def build_upper(g, m, fu, gate_level, wall=GRAY_WALL):
    """Upper-floor part: gated corridor and the hole in the floor."""
    for b in range(-1, 7):
        for a in range(-1, 3):
            x, z = m.xz(a, b)
            for y in range(fu + 1, fu + 8):
                interior = 0 <= a <= 1 and 0 <= b <= 5
                if interior:
                    g.set(x, y, z, AIR)
                elif b == -1 and 0 <= a <= 1:
                    # Doorway: bars up to 3 high when gated.
                    if y <= fu + 3:
                        g.set(x, y, z, BARS_EW if gate_level else AIR, force=True)
                    else:
                        g.set(x, y, z, wall, force=True)
                elif not is_wall(g.get(x, y, z)[0]):
                    g.set(x, y, z, wall)
    # Keycard reader beside the door, facing the approach side.
    if gate_level:
        x, z = m.xz(2, -1)
        g.set(x, fu + 2, z, (reader_id(gate_level, 'left'), {}), force=True)
    # Hole in the floor above flight 1.
    for b in range(1, 6):
        for a in range(0, 2):
            x, z = m.xz(a, b)
            g.set(x, fu, z, AIR, force=True)
    x, z = m.xz(1, 2)
    g.set(x, fu + 7, z, LIGHT)


def build_lower(g, m, fu, wall=GRAY_WALL, fill=METAL):
    """Lower-floor part: stairs, landing and the arrival enclosure."""
    fl = fu - FLOOR_H
    down = m.dir_of_b(-1)  # flight 1 rises towards -b
    up = m.dir_of_b(1)     # flight 2 rises towards +b
    for b in range(-1, 9):
        for a in range(-1, 5):
            x, z = m.xz(a, b)
            for y in range(fl + 1, fl + 8):
                if 0 <= a <= 3 and 0 <= b <= 7:
                    g.set(x, y, z, AIR)
                elif b == -1 and 2 <= a <= 3 and y <= fl + 3:
                    g.set(x, y, z, AIR, force=True)  # open exit
                elif not is_wall(g.get(x, y, z)[0]):
                    g.set(x, y, z, wall)
    # Flight 1 (a0-1): b1..5, top of b1 sits in the lower floor's ceiling.
    for b in range(1, 6):
        sy = fu - b
        for a in range(0, 2):
            x, z = m.xz(a, b)
            g.set(x, sy, z, stairs(down), force=True)
            for y in range(sy + 1, fu):
                g.set(x, y, z, AIR, force=True)  # ceiling cut / headroom
            for y in range(fl + 1, sy):
                g.set(x, y, z, fill)
    # Landing (a0-3, b6-7).
    for b in range(6, 8):
        for a in range(0, 4):
            x, z = m.xz(a, b)
            g.set(x, fu - 5, z, TILES, force=True)
            for y in range(fl + 1, fu - 5):
                g.set(x, y, z, fill)
    # Flight 2 (a2-3): b5..3 back down to the lower floor.
    for i, b in enumerate(range(5, 2, -1)):
        sy = fu - 6 - i
        for a in range(2, 4):
            x, z = m.xz(a, b)
            g.set(x, sy, z, stairs(up), force=True)
            for y in range(fl + 1, sy):
                g.set(x, y, z, fill)
    x, z = m.xz(3, 0)
    g.set(x, fl + 7, z, LIGHT)


def seal(g, x, z_range, y_range, block=METAL):
    for z in z_range:
        for y in y_range:
            g.set(x, y, z, block, force=True)


# ---------------------------------------------------------------------------
# Deep-lab prefab remapping: gate levels and loot per room.
# Original gate readers sit at x = 7, 19, 31, 39 (gates g1..g4); rooms R1..R4
# are the north rooms between them. Coordinates below are ORIGINAL prefab
# coordinates, applied before mirroring.
# ---------------------------------------------------------------------------

READER_X = {7: 0, 19: 1, 31: 2, 39: 3}


def deep_remap(gates, rooms):
    def room_of(x):
        if x < 20:
            return 0
        if x < 32:
            return 1
        if x < 40:
            return 2
        return 3

    def remap(name, props, nbt, pos):
        x, y, z = pos
        if name.startswith('scp_additions:') and name.endswith('_reader'):
            side = 'left' if 'left' in name else 'right'
            name = reader_id(gates[READER_X[x]], side)
        elif name == 'minecraft:chest' and nbt is not None and 'LootTable' in nbt:
            nbt = tag.Compound(dict(nbt))
            nbt['LootTable'] = tag.String(f'deceasedcraft:chests/scp_lab/{rooms[room_of(x)]}')
        return name, props, nbt

    return remap


# ---------------------------------------------------------------------------
# Jigsaw + export
# ---------------------------------------------------------------------------

def jigsaw(g, x, y, z, facing, final_block, name, target, pool):
    orient = f'{facing}_north'
    nbt = tag.Compound({
        'id': tag.String('minecraft:jigsaw'),
        'name': tag.String(name),
        'target': tag.String(target),
        'pool': tag.String(pool),
        'joint': tag.String('rigid'),
        'final_state': tag.String(final_block),
    })
    g.set(x, y, z, ('minecraft:jigsaw', {'orientation': orient}), nbt=nbt, force=True)


def export(g, path, box):
    (x0, y0, z0), (x1, y1, z1) = box
    palette, index, blocks = [], {}, []
    for y in range(y0, y1 + 1):
        for z in range(z0, z1 + 1):
            for x in range(x0, x1 + 1):
                name, props, nbt = g.get(x, y, z)
                key = (name, tuple(sorted(props.items())))
                if key not in index:
                    index[key] = len(palette)
                    entry = {'Name': tag.String(name)}
                    if props:
                        entry['Properties'] = tag.Compound({k: tag.String(v) for k, v in props.items()})
                    palette.append(tag.Compound(entry))
                blk = {
                    'pos': tag.List[tag.Int]([x - x0, y - y0, z - z0]),
                    'state': tag.Int(index[key]),
                }
                if nbt is not None:
                    blk['nbt'] = nbt
                blocks.append(tag.Compound(blk))
    root = tag.Compound({
        'size': tag.List[tag.Int]([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1]),
        'entities': tag.List[tag.Compound]([]),
        'blocks': tag.List[tag.Compound](blocks),
        'palette': tag.List[tag.Compound](palette),
        'DataVersion': tag.Int(DATA_VERSION),
    })
    nbtlib.File(root, root_name='').save(path, gzipped=True)
    print(f'  wrote {os.path.relpath(path)}  size={x1 - x0 + 1}x{y1 - y0 + 1}x{z1 - z0 + 1}  palette={len(palette)}')


def floor_range(n):
    return range(n * FLOOR_H, n * FLOOR_H + FLOOR_H)


# ---------------------------------------------------------------------------
# General lab: 1F (surface) -> B1 -> B2
# ---------------------------------------------------------------------------

def build_general():
    g = Grid('scp_lab')
    tier1 = os.path.join(SRC, 'scp_lab_tier1.nbt')
    deep = os.path.join(SRC, 'scp_deep_lab.nbt')

    # Stack (combined Y): B2 0-8, B1 9-17, 1F 18-28.
    g.load_floor(deep, 0, 'rot180', range(0, 9),
                 deep_remap(gates=[3, 3, 3, 3], rooms=['containment', 'archive', 'containment', 'lab_vault']))
    g.load_floor(deep, 9, 'mirror_z', range(0, 9),
                 deep_remap(gates=[2, 2, 2, 3], rooms=['archive', 'archive', 'containment', 'containment']))
    g.load_floor(tier1, 18, 'none', range(0, 11))

    # Seal the prefab's open corridor end on the underground floors.
    seal(g, 46, range(21, 26), range(1, 8))    # B2 (rot180 moves x0 -> x46)
    seal(g, 0, range(21, 26), range(10, 17))   # B1

    # 1F -> B1: SW corner, gate on the hall wall (z26), L2.
    m1 = Module(ox=1, oz=27, ax=1, bz=1)
    build_upper(g, m1, fu=18, gate_level=2, wall=('scp_the_facility:foundation_wall', {}))
    build_lower(g, m1, fu=18)

    # B1 -> B2: SE far room of B1, L3.
    m2 = Module(ox=45, oz=27, ax=-1, bz=1)
    build_upper(g, m2, fu=9, gate_level=3)
    build_lower(g, m2, fu=9)

    # Jigsaw pair joining the surface piece to the underground piece.
    jigsaw(g, 23, 18, 23, 'down', 'scp_the_facility:foundation_floor_tiles',
           'deceasedcraft:scp_lab_down', 'deceasedcraft:scp_lab_up', 'deceasedcraft:scp_lab/levels')
    jigsaw(g, 23, 17, 23, 'up', 'scp_the_facility:foundation_ceiling',
           'deceasedcraft:scp_lab_up', 'minecraft:empty', 'minecraft:empty')

    for w in g.warnings:
        print('  WARN', w)
    export(g, os.path.join(OUT, 'scp_lab_surface.nbt'), ((0, 18, 0), (46, 28, 46)))
    export(g, os.path.join(OUT, 'scp_lab_levels.nbt'), ((0, 0, 0), (46, 17, 46)))
    return g


# ---------------------------------------------------------------------------
# Deep lab: entrance hut -> ladder shaft -> B1 -> B2 -> B3
# ---------------------------------------------------------------------------

LADDER_X, LADDER_Z = 3, 21
SHAFT_Y0, SHAFT_H = 27, 6
HUT_Y0 = SHAFT_Y0 + SHAFT_H          # 33
HUT_X0, HUT_Z0 = LADDER_X - 4, LADDER_Z - 2
LADDER = ('minecraft:ladder', {'facing': 'south', 'waterlogged': 'false'})


def build_hut(g, gate_level):
    wall = GRAY_WALL
    for hx in range(9):
        for hz in range(9):
            x, z = HUT_X0 + hx, HUT_Z0 + hz
            g.set(x, HUT_Y0, z, TILES, force=True)
            g.set(x, HUT_Y0 + 6, z, METAL, force=True)
            edge = hx in (0, 8) or hz in (0, 8)
            for hy in range(1, 6):
                g.set(x, HUT_Y0 + hy, z, wall if edge else AIR, force=True)
    # Entrance on the south side.
    for hx in range(3, 6):
        for hy in range(1, 4):
            g.set(HUT_X0 + hx, HUT_Y0 + hy, HUT_Z0 + 8, AIR, force=True)
    # Gated partition between the lobby and the shaft room.
    for hx in range(1, 8):
        for hy in range(1, 6):
            gated = 3 <= hx <= 5 and hy <= 3
            g.set(HUT_X0 + hx, HUT_Y0 + hy, HUT_Z0 + 4, BARS_EW if gated else wall, force=True)
    g.set(HUT_X0 + 6, HUT_Y0 + 2, HUT_Z0 + 4, (reader_id(gate_level, 'left'), {}), force=True)
    g.set(HUT_X0 + 4, HUT_Y0 + 5, HUT_Z0 + 6, LIGHT, force=True)
    g.set(HUT_X0 + 4, HUT_Y0 + 5, HUT_Z0 + 2, LIGHT, force=True)
    g.set(HUT_X0 + 1, HUT_Y0 + 4, HUT_Z0 + 1, ('scp_the_facility:alarm_on', {}), force=True)
    # Ladder hole in the hut floor.
    g.set(LADDER_X, HUT_Y0, LADDER_Z, LADDER, force=True)


def build_shaft(g):
    for y in range(SHAFT_Y0, SHAFT_Y0 + SHAFT_H):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                g.set(LADDER_X + dx, y, LADDER_Z + dz, METAL, force=True)
        g.set(LADDER_X, y, LADDER_Z, LADDER, force=True)


def build_deep():
    g = Grid('scp_deep_lab')
    deep = os.path.join(SRC, 'scp_deep_lab.nbt')

    # Stack (combined Y): B3 0-8, B2 9-17, B1 18-26.
    g.load_floor(deep, 0, 'none', range(0, 9),
                 deep_remap(gates=[6, 6, 6, 6], rooms=['clearance_6', 'clearance_6', 'deep_vault', 'deep_vault']))
    g.load_floor(deep, 9, 'mirror_x', range(0, 9),
                 deep_remap(gates=[5, 5, 5, 5], rooms=['clearance_5', 'clearance_6', 'clearance_6', 'clearance_6']))
    g.load_floor(deep, 18, 'none', range(0, 9),
                 deep_remap(gates=[3, 4, 4, 4], rooms=['clearance_4', 'clearance_4', 'clearance_5', 'clearance_5']))

    seal(g, 0, range(21, 26), range(1, 8))     # B3
    seal(g, 46, range(21, 26), range(10, 17))  # B2 (mirror_x)
    seal(g, 0, range(21, 26), range(19, 26))   # B1

    # B1 -> B2: NE far room, L5.
    m1 = Module(ox=45, oz=19, ax=-1, bz=-1)
    build_upper(g, m1, fu=18, gate_level=5)
    build_lower(g, m1, fu=18)
    # B2 -> B3: NW far room (B2 is mirrored), L6.
    m2 = Module(ox=1, oz=19, ax=1, bz=-1)
    build_upper(g, m2, fu=9, gate_level=6)
    build_lower(g, m2, fu=9)

    # Ladder into B1: through the ceiling at the start of the corridor.
    for y in range(19, 27):
        g.set(LADDER_X, y, LADDER_Z, LADDER, force=True)

    build_shaft(g)
    build_hut(g, gate_level=3)

    # hut -> shaft
    jigsaw(g, LADDER_X - 1, HUT_Y0, LADDER_Z, 'down', TILES[0],
           'deceasedcraft:scp_deep_lab_hut_down', 'deceasedcraft:scp_deep_lab_shaft_up', 'deceasedcraft:scp_deep_lab/shaft')
    jigsaw(g, LADDER_X - 1, HUT_Y0 - 1, LADDER_Z, 'up', METAL[0],
           'deceasedcraft:scp_deep_lab_shaft_up', 'minecraft:empty', 'minecraft:empty')
    # shaft -> levels
    jigsaw(g, LADDER_X + 1, SHAFT_Y0, LADDER_Z, 'down', METAL[0],
           'deceasedcraft:scp_deep_lab_shaft_down', 'deceasedcraft:scp_deep_lab_levels_up', 'deceasedcraft:scp_deep_lab/levels')
    jigsaw(g, LADDER_X + 1, SHAFT_Y0 - 1, LADDER_Z, 'up', CEILING[0],
           'deceasedcraft:scp_deep_lab_levels_up', 'minecraft:empty', 'minecraft:empty')

    for w in g.warnings:
        print('  WARN', w)
    export(g, os.path.join(OUT, 'scp_deep_lab_entrance.nbt'),
           ((HUT_X0, HUT_Y0, HUT_Z0), (HUT_X0 + 8, HUT_Y0 + 6, HUT_Z0 + 8)))
    export(g, os.path.join(OUT, 'scp_deep_lab_shaft.nbt'),
           ((LADDER_X - 1, SHAFT_Y0, LADDER_Z - 1), (LADDER_X + 1, SHAFT_Y0 + SHAFT_H - 1, LADDER_Z + 1)))
    export(g, os.path.join(OUT, 'scp_deep_lab_levels.nbt'), ((0, 0, 0), (46, 26, 46)))
    return g


if __name__ == '__main__':
    print('General lab (1F/B1/B2):')
    gen = build_general()
    print('Deep lab (entrance/B1/B2/B3):')
    dp = build_deep()
    if '--dump' in sys.argv:
        import pickle
        with open(os.path.join(HERE, '.grids.pickle'), 'wb') as f:
            pickle.dump({'scp_lab': gen.cells, 'scp_deep_lab': dp.cells}, f)
