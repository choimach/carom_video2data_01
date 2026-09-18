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

from src.physics import spin
from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM

BALL_DIAMETER_MM = 61.5
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0

# Measured across five matches; the tools/measure_physics.py commit carries the
# sample sizes and how each was arrived at.
CUSHION_RESTITUTION = 0.69      # of the speed across the rail, 1315 bounces
CUSHION_TANGENTIAL = 0.92       # of the speed along it, with no side on the ball

# A ball does not leave a cushion at the angle it arrived. It opens up, most of
# all when it comes in shallow: measured over 1297 bounces, a ball arriving 18
# degrees off the rail's normal leaves at 34, one arriving at 46 leaves at 55,
# and one arriving at 79 leaves at 80. Scaling the two components by the
# coefficients above accounts for about three of those degrees; the rest is the
# ball's own roll driving it along the rail after the bounce, the same effect
# that carries a cue ball forward through a contact.
#
# Against the component scaling it replaces, on the same 411 plays, this is
# worth less than the physics of it suggests: the same 107 mm at the first
# rail, the same half of the plays within one cushion, and 440 mm of drift at
# two seconds against 473. It is kept because it is what the table does and the
# components cannot produce it, not because it rescued the simulator. What
# dominates the error happens before the first cushion.
#
# Angles are signed along the direction the ball was already travelling, which
# matters more than it sounds: measuring them as magnitudes folds every rebound
# that came back short into one that went long, and inflated the opening at
# near-square incidence from 10 degrees to 19. Symmetry fixes the first entry -
# a ball arriving exactly square has no direction to open into - and the rest
# is the measurement.
REBOUND_IN = np.array([0.0, 6.7, 17.8, 27.2, 38.0, 46.4, 56.0, 66.1, 79.3, 90.0])
REBOUND_OUT = np.array([0.0, 16.7, 33.7, 41.8, 50.1, 55.4, 63.0, 70.6, 80.1, 90.0])
REBOUND_SPEED = np.array([0.817, 0.817, 0.844, 0.829, 0.824, 0.818, 0.808, 0.835, 0.912, 0.912])
BALL_RESTITUTION = 0.944        # 322 contacts, run back to the moment of contact
# A frictionless cue ball leaves a cut square to the line joining the centres
# and no part of it carries on. A real one does, because it arrives rolling and
# the roll survives the contact: measured over 297 collisions it keeps 0.089 of
# the speed it came in with, going forward along those centres, and it does so
# at every cut angle from a quarter ball to nearly full. Leaving this out is
# what put the simulated cue ball 658 mm from the real one by the first rail
# after a contact, against 85 mm for a ball that reached the rail untouched.
CUE_CARRY = 0.089

# Deceleration in mm/s^2 against speed in mm/s, fitted to how far a ball
# actually runs rather than differenced off the tracks.
#
# Differencing consecutive frames measures centroid noise, not deceleration: it
# read 6,584 mm/s^2 at walking pace, which would stop a ball in a thirtieth of
# a second. Measuring over longer windows gave a curve four times shallower
# than the table - a cue ball opening at the 2699 mm/s professionals average
# ran 14,150 mm against the 5,680 mm the video records.
#
# So the shape is the measurement and the scale is fitted to distance: 1.8x at
# walking pace rising to 6x at speed, which lands within 7% across four bands of
# object-ball launch speed and gives 5,329 mm for that median cue opening.
#
# It cost the sweep its coarse step. Scoring windows are a quarter of a degree
# wide, and on the old slippery cloth a ball that missed still wandered into
# something, so a 2-degree sweep found lines. Now it has to look properly:
# 2 degrees finds 3 lines on a layout where a quarter of a degree finds 36.
DECAY_SPEED = np.array([0.0, 300.0, 600.0, 1000.0, 1600.0, 2400.0, 12000.0])
DECAY_RATE = np.array([104.0, 113.0, 163.0, 284.0, 594.0, 1968.0, 2634.0])

REST_SPEED_MM_S = 12.0          # below this a ball has stopped
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
        edge = BALL_RADIUS_MM

        if x < edge:
            x, vx, rail = edge + (edge - x), -vx, "left"
        elif x > self.length - edge:
            x, vx, rail = (self.length - edge) - (x - (self.length - edge)), -vx, "right"
        if y < edge:
            y, vy, rail = edge + (edge - y), -vy, "top"
        elif y > self.width - edge:
            y, vy, rail = (self.width - edge) - (y - (self.width - edge)), -vy, "bottom"
        if rail is None:
            return position, velocity, None

        # vx, vy already point away from the rail. Split them into the part
        # across it and the part along it, turn the angle by what the table
        # actually does, and put them back.
        if rail in ("left", "right"):
            across, along = vx, vy
        else:
            across, along = vy, vx
        speed = float(np.hypot(across, along))
        if speed <= 0:
            return np.array([x, y]), np.array([vx, vy]), rail

        incoming = np.degrees(np.arctan2(abs(along), abs(across)))
        outgoing = float(np.interp(incoming, REBOUND_IN, REBOUND_OUT))
        outgoing = min(89.0, outgoing + side_degrees)
        kept = float(np.interp(incoming, REBOUND_IN, REBOUND_SPEED))

        speed *= kept
        across_out = np.sign(across) * speed * np.cos(np.radians(outgoing))
        along_out = (np.sign(along) if along != 0 else 1.0) * speed * np.sin(np.radians(outgoing))
        if rail in ("left", "right"):
            vx, vy = across_out, along_out
        else:
            vy, vx = across_out, along_out
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


def _clearance(position, colours):
    """How far the closest ball is from touching a rail or another ball."""
    room = 10 ** 6
    for index, colour in enumerate(colours):
        x, y = position[colour]
        room = min(room, x - BALL_RADIUS_MM, y - BALL_RADIUS_MM,
                   TABLE_LENGTH_MM - BALL_RADIUS_MM - x, TABLE_WIDTH_MM - BALL_RADIUS_MM - y)
        for other in colours[index + 1:]:
            gap = position[other] - position[colour]
            room = min(room, float(np.hypot(*gap)) - BALL_DIAMETER_MM)
    return max(0.0, room)


def simulate_with_spin(layout, cue, velocity, tips_side=0.0, tips_vertical=0.0,
                       fps=60.0, max_seconds=20.0, substep_mm=4.0):
    """The same stroke, played by balls that carry spin.

    `tips_side` and `tips_vertical` are 당점 in the units a player is given -
    tips from centre, three being the edge before a miscue. Everything else
    matches `simulate`, so the two can be compared shot for shot.
    """
    colours = list(layout)
    heading = np.asarray(velocity, dtype=float)
    speed = float(np.hypot(*heading))
    balls = {c: spin.Ball(layout[c]) for c in colours}
    balls[cue] = spin.struck(layout[cue], heading, speed,
                             tips_side=tips_side, tips_vertical=tips_vertical)

    frame_gap = 1.0 / fps
    paths = {c: [balls[c].position.copy()] for c in colours}
    events = []
    clock, next_frame = 0.0, frame_gap

    while clock < max_seconds:
        fastest = max(ball.speed for ball in balls.values())
        if fastest < REST_SPEED_MM_S:
            break
        positions = {c: balls[c].position for c in colours}
        clearance = _clearance(positions, colours)
        step = min(max(clearance * 0.5, substep_mm) / fastest, frame_gap)

        for ball in balls.values():
            spin.roll_on(ball, step)

        frame_index = int(round(clock / frame_gap))
        for colour, ball in balls.items():
            rail = _rail_reached(ball.position)
            if rail is None:
                continue
            ball.position = _inside(ball.position)
            spin.bounce(ball, rail)
            if colour == cue:
                events.append((frame_index, "cushion", rail))

        for i, first in enumerate(colours):
            for second in colours[i + 1:]:
                gap = balls[second].position - balls[first].position
                distance = float(np.hypot(*gap))
                if distance >= BALL_DIAMETER_MM or distance == 0:
                    continue
                push = gap / distance * ((BALL_DIAMETER_MM - distance) / 2.0)
                balls[first].position = balls[first].position - push
                balls[second].position = balls[second].position + push
                if float(np.dot(balls[first].velocity - balls[second].velocity, gap)) > 0:
                    spin.collide(balls[first], balls[second])
                else:
                    spin.collide(balls[second], balls[first])
                if cue in (first, second):
                    events.append((frame_index, "ball", second if first == cue else first))

        clock += step
        if clock >= next_frame:
            for colour in colours:
                paths[colour].append(balls[colour].position.copy())
            next_frame += frame_gap

    settled = max(ball.speed for ball in balls.values()) < REST_SPEED_MM_S
    return Shot({c: np.array(p) for c, p in paths.items()}, events, fps, settled)


def _rail_reached(position, length=TABLE_LENGTH_MM, width=TABLE_WIDTH_MM):
    if position[0] < BALL_RADIUS_MM:
        return "left"
    if position[0] > length - BALL_RADIUS_MM:
        return "right"
    if position[1] < BALL_RADIUS_MM:
        return "top"
    if position[1] > width - BALL_RADIUS_MM:
        return "bottom"
    return None


def _inside(position, length=TABLE_LENGTH_MM, width=TABLE_WIDTH_MM):
    return np.array([
        min(max(position[0], BALL_RADIUS_MM), length - BALL_RADIUS_MM),
        min(max(position[1], BALL_RADIUS_MM), width - BALL_RADIUS_MM),
    ])


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
        # Step by how much room there is, not by a fixed distance. A ball in
        # open table can cross hundreds of millimetres before anything can
        # happen to it, and checking every four is most of the cost of a shot.
        # The limit is the nearest thing it could reach: a rail, or the surface
        # of another ball.
        clearance = _clearance(position, colours)
        step = min(max(clearance * 0.5, substep_mm) / fastest, frame_gap)

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
