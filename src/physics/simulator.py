"""A carom table that can be played on without a camera.

Every constant here was measured from the matches in data/scans rather than
looked up - see tools/measure_physics.py - and the units are the dataset's own:
millimetres and seconds, with the nose line running from (0, 0) to
(2844, 1422).

What is modelled is translation. A ball's speed decays along the curve the
broadcasts show, a cushion returns the measured fractions of the speed across
and along it, and two balls exchange momentum along the line joining their
centres. What is not modelled is spin as a state of its own: the slide-to-roll
transition is folded into the decay curve, where it shows as a ball losing
speed four times faster at two metres a second than at a quarter of one, and
side is an argument to the cushion rather than something a ball carries. That
is enough to reproduce a shot's path and not enough to reproduce a masse.
"""

import numpy as np

from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM

BALL_DIAMETER_MM = 61.5
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0

# Measured across five matches; the tools/measure_physics.py commit carries the
# sample sizes and how each was arrived at.
CUSHION_RESTITUTION = 0.69      # of the speed across the rail, 1315 bounces
CUSHION_TANGENTIAL = 0.92       # of the speed along it, with no side on the ball
BALL_RESTITUTION = 0.944        # 322 contacts, run back to the moment of contact
# A frictionless cue ball leaves a cut square to the line joining the centres
# and no part of it carries on. A real one does, because it arrives rolling and
# the roll survives the contact: measured over 297 collisions it keeps 0.089 of
# the speed it came in with, going forward along those centres, and it does so
# at every cut angle from a quarter ball to nearly full. Leaving this out is
# what put the simulated cue ball 658 mm from the real one by the first rail
# after a contact, against 85 mm for a ball that reached the rail untouched.
CUE_CARRY = 0.089

# Deceleration in mm/s^2 against speed in mm/s. A ball sheds speed four times
# faster at two metres a second than at a quarter of one, because it is still
# sliding rather than rolling; the curve is the measurement, not a model of it.
DECAY_SPEED = np.array([0.0, 350.0, 700.0, 1200.0, 1950.0, 3000.0, 12000.0])
DECAY_RATE = np.array([46.0, 46.0, 51.0, 74.0, 216.0, 439.0, 439.0])

REST_SPEED_MM_S = 12.0          # below this a ball has stopped
# Side on the ball widens or narrows a rebound. Measured on 338 cushions:
# reverse side keeps 0.35 of the speed along the rail where neutral keeps 0.92
# and running keeps 1.17, so a degree off the mirror angle is worth about this
# much of the tangential speed.
SIDE_PER_DEGREE = 0.020

RAILS = ("left", "right", "top", "bottom")


def deceleration(speed):
    """How fast a ball at this speed is losing it, in mm/s^2."""
    return float(np.interp(speed, DECAY_SPEED, DECAY_RATE))


class Table:
    """The playing surface, and the rules for meeting its edges."""

    def __init__(self, length=TABLE_LENGTH_MM, width=TABLE_WIDTH_MM,
                 restitution=CUSHION_RESTITUTION, tangential=CUSHION_TANGENTIAL):
        self.length = length
        self.width = width
        self.restitution = restitution
        self.tangential = tangential

    def bounce(self, position, velocity, side_degrees=0.0):
        """Reflect a ball that has reached a rail. Returns (position, velocity, rail)."""
        x, y = position
        vx, vy = velocity
        rail = None
        low, high = BALL_RADIUS_MM, None

        if x < low:
            x, vx, rail = low + (low - x), -vx, "left"
        elif x > self.length - low:
            x, vx, rail = (self.length - low) - (x - (self.length - low)), -vx, "right"
        if y < low:
            y, vy, rail = low + (low - y), -vy, "top"
        elif y > self.width - low:
            y, vy, rail = (self.width - low) - (y - (self.width - low)), -vy, "bottom"
        if rail is None:
            return position, velocity, None

        across = self.restitution
        along = self.tangential * (1.0 + SIDE_PER_DEGREE * side_degrees)
        along = max(0.0, along)
        if rail in ("left", "right"):
            vx, vy = vx * across, vy * along
        else:
            vx, vy = vx * along, vy * across
        return np.array([x, y]), np.array([vx, vy]), rail


def collide(position_a, velocity_a, position_b, velocity_b, restitution=BALL_RESTITUTION,
            carry=CUE_CARRY):
    """Two equal balls meeting: they trade speed along the line of centres.

    Momentum crosses that line and nothing crosses the other way, which is the
    frictionless approximation the thickness measurement rests on too - it is
    what sends a struck ball along the centres and leaves the striking ball
    square to them.

    `carry` is where this stops being frictionless. A ball arrives rolling, and
    the roll does not stop because the ball in front of it did, so the striker
    keeps a little of its speed going forward along the centres rather than
    leaving square. The measured 0.089 is small and it is not optional: without
    it every path that passes through a contact bends the wrong way.
    """
    line = position_b - position_a
    distance = float(np.hypot(*line))
    if distance == 0:
        return velocity_a, velocity_b
    normal = line / distance
    along_a = float(np.dot(velocity_a, normal))
    along_b = float(np.dot(velocity_b, normal))
    if along_a - along_b <= 0:
        return velocity_a, velocity_b       # already separating
    share = (1.0 + restitution) / 2.0
    transfer = share * (along_a - along_b)
    new_a = along_a - transfer + carry * abs(along_a)
    new_b = along_b + transfer
    return (velocity_a + (new_a - along_a) * normal,
            velocity_b + (new_b - along_b) * normal)


class Shot:
    """The outcome of one simulated stroke."""

    def __init__(self, paths, events, fps, settled):
        self.paths = paths          # {colour: (n, 2) array of mm}
        self.events = events        # [(frame, kind, detail)] in the order they happened
        self.fps = fps
        self.settled = settled      # False when the balls were still moving at the end

    @property
    def cushions(self):
        return [e for e in self.events if e[1] == "cushion"]

    def scored(self, cue):
        """Three cushions before the second object ball, which is how a point is made.

        Four, five or six cushions score just the same - what the rule requires
        is three before the second ball, not exactly three.
        """
        touched, cushions = [], 0
        for _frame, kind, detail in self.events:
            if kind == "cushion":
                cushions += 1
            elif detail not in touched:
                touched.append(detail)
                if len(touched) == 2:
                    return cushions >= 3
        return False

    def __repr__(self):
        return (f"<Shot {len(self.events)} events, "
                f"{len(self.cushions)} cushions, {len(next(iter(self.paths.values())))} frames>")


def simulate(layout, cue, velocity, table=None, fps=60.0, max_seconds=20.0,
             substep_mm=4.0, side_degrees=0.0):
    """Play one stroke and record where every ball went.

    `layout` maps colour to a starting position in table millimetres, `cue`
    names the ball being struck and `velocity` is its opening velocity in mm/s.
    Positions come back sampled at `fps`, so a simulated play can be compared
    against a tracked one frame for frame.

    The step is chosen from the fastest ball rather than fixed, so that nothing
    moves more than a few millimetres between checks: a cue ball at four metres
    a second crosses a whole ball in a sixtieth of a second, and a fixed step
    that size would let it pass clean through an object ball.
    """
    table = table or Table()
    colours = list(layout)
    position = {c: np.array(layout[c], dtype=float) for c in colours}
    speed = {c: np.zeros(2) for c in colours}
    speed[cue] = np.array(velocity, dtype=float)

    frame_gap = 1.0 / fps
    paths = {c: [position[c].copy()] for c in colours}
    events = []
    clock, next_frame = 0.0, frame_gap

    while clock < max_seconds:
        fastest = max(float(np.hypot(*speed[c])) for c in colours)
        if fastest < REST_SPEED_MM_S:
            break
        step = min(substep_mm / fastest, frame_gap)

        for colour in colours:
            velocity_now = speed[colour]
            magnitude = float(np.hypot(*velocity_now))
            if magnitude < REST_SPEED_MM_S:
                speed[colour] = np.zeros(2)
                continue
            position[colour] = position[colour] + velocity_now * step
            slowed = max(0.0, magnitude - deceleration(magnitude) * step)
            speed[colour] = velocity_now * (slowed / magnitude)

        frame_index = int(round(clock / frame_gap))
        for colour in colours:
            moved, changed, rail = table.bounce(position[colour], speed[colour], side_degrees)
            if rail is not None:
                position[colour], speed[colour] = moved, changed
                if colour == cue:
                    events.append((frame_index, "cushion", rail))

        for i, first in enumerate(colours):
            for second in colours[i + 1:]:
                gap = position[second] - position[first]
                distance = float(np.hypot(*gap))
                if distance >= BALL_DIAMETER_MM or distance == 0:
                    continue
                # Separate them before resolving, or they collide again next step.
                overlap = (BALL_DIAMETER_MM - distance) / 2.0
                push = gap / distance * overlap
                position[first] = position[first] - push
                position[second] = position[second] + push
                speed[first], speed[second] = collide(
                    position[first], speed[first], position[second], speed[second])
                if cue in (first, second):
                    other = second if first == cue else first
                    events.append((frame_index, "ball", other))

        clock += step
        while clock >= next_frame:
            for colour in colours:
                paths[colour].append(position[colour].copy())
            next_frame += frame_gap

    settled = all(float(np.hypot(*speed[c])) < REST_SPEED_MM_S for c in colours)
    return Shot({c: np.array(v) for c, v in paths.items()}, events, fps, settled)
