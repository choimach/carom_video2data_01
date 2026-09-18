"""A ball that carries spin, and what spin does to it.

The simulator next door models translation and folds everything spin does into
measured curves: a decay that stands for the slide-to-roll transition, and a
cushion that takes side as an argument rather than reading it off the ball. That
is enough to draw a route and not enough to say 당점 - which is the thing a
player is actually told, in tips and clock positions.

So this carries the state the table needs:

* `slip` - the velocity of the point of the ball touching the cloth, which is
  what friction acts on. Zero means the ball is rolling. A struck ball slides
  first and the slide is where most of its speed goes.
* `side` - spin about the vertical axis, in radians per second. This is what a
  player puts on with 1시 or 9시 당점, what a cushion converts into a wider or
  narrower rebound, and what throws an object ball off the line of centres.

Units are the dataset's own: millimetres, seconds, radians per second.

The constants are the standard ones for billiard cloth, except where this
project has measured its own. What holds them in place: a cue ball opening at
2699 mm/s - the median over 1596 tracked plays - must travel 5680 mm.
"""

import numpy as np

BALL_DIAMETER_MM = 61.5
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0
GRAVITY_MM_S2 = 9810.0

# Cloth. Sliding is where a struck ball sheds speed; rolling barely slows at
# all, which is why a cue ball crosses the table three or four times.
SLIDING_FRICTION = 0.20
ROLLING_FRICTION = 0.0106
# Spin about the vertical axis dies on its own, slowly: it has only the contact
# patch to work against. Slowly enough that over the first second - as far as
# the paths here are compared - the value makes no difference at all: 0.01,
# 0.022 and 0.04 give the same answer to the millimetre. It is left at the low
# end, and whatever settles it will have to be a shot measured further out.
SPIN_FRICTION = 0.01

# A cue tip is 12 mm across and the ball 61.5, so the furthest a player can
# strike from centre before miscuing is about half the radius. Three tips is
# that edge, which makes a tip five millimetres.
TIP_MM = 5.0
MAX_TIPS = 3.0


class Ball:
    """Where a ball is, how it is moving, and how it is spinning."""

    def __init__(self, position, velocity=(0.0, 0.0), side=0.0, slip=None):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.side = float(side)
        # A ball struck in the middle leaves sliding: its contact point is
        # moving as fast as the ball is.
        self.slip = (np.asarray(slip, dtype=float) if slip is not None
                     else self.velocity.copy())

    @property
    def speed(self):
        return float(np.hypot(*self.velocity))

    @property
    def rolling(self):
        return float(np.hypot(*self.slip)) < 1.0

    def copy(self):
        return Ball(self.position.copy(), self.velocity.copy(), self.side,
                    self.slip.copy())


def struck(position, direction, speed, tips_side=0.0, tips_vertical=0.0):
    """A ball just struck: where the tip landed decides what it carries.

    `tips_side` is positive for right-hand side (3시 쪽), `tips_vertical`
    positive for follow (12시) and negative for draw (6시). Both are in tips,
    capped at three, which is where a cue starts to miscue.

    A tip `b` millimetres off centre puts 5·v·b / 2R² of spin on the ball -
    the standard result for a thin impulse on a sphere.
    """
    direction = np.asarray(direction, dtype=float)
    length = float(np.hypot(*direction))
    if length == 0:
        raise ValueError("a struck ball needs a direction")
    heading = direction / length
    velocity = heading * speed

    sideways = np.clip(tips_side, -MAX_TIPS, MAX_TIPS) * TIP_MM
    vertical = np.clip(tips_vertical, -MAX_TIPS, MAX_TIPS) * TIP_MM
    side = 5.0 * speed * sideways / (2.0 * BALL_RADIUS_MM ** 2)

    # Follow and draw show up in how fast the contact point is moving: a ball
    # struck high is already part-way to rolling, one struck low is spinning
    # backwards under itself.
    roll_fraction = np.clip(vertical / (0.4 * BALL_RADIUS_MM), -1.5, 1.0)
    slip = velocity * (1.0 - roll_fraction)
    return Ball(position, velocity, side, slip)


def roll_on(ball, seconds):
    """Carry a ball forward, sliding first and then rolling."""
    if ball.speed < 1e-9 and abs(ball.side) < 1e-9:
        return
    ball.position = ball.position + ball.velocity * seconds

    slip_speed = float(np.hypot(*ball.slip))
    if slip_speed > 1.0:
        # Sliding. Friction acts against the contact point, slowing the ball and
        # killing the slip seven halves as fast - which is what turns a slide
        # into a roll without anything having to decide when.
        direction = ball.slip / slip_speed
        ball.velocity = ball.velocity - direction * SLIDING_FRICTION * GRAVITY_MM_S2 * seconds
        shed = 3.5 * SLIDING_FRICTION * GRAVITY_MM_S2 * seconds
        ball.slip = direction * max(0.0, slip_speed - shed)
    else:
        speed = ball.speed
        if speed > 1e-9:
            direction = ball.velocity / speed
            ball.velocity = direction * max(0.0, speed - ROLLING_FRICTION * GRAVITY_MM_S2 * seconds)
        ball.slip = np.zeros(2)

    # Side spin has only the contact patch to work against, so it outlives the
    # roll and is still there at the third cushion.
    decay = SPIN_FRICTION * GRAVITY_MM_S2 / BALL_RADIUS_MM * seconds
    if abs(ball.side) <= decay:
        ball.side = 0.0
    else:
        ball.side -= np.sign(ball.side) * decay


def tips_of(ball):
    """The side on a ball, back in the units a player was told: tips."""
    if ball.speed < 1e-9:
        return 0.0
    tips = ball.side * 2.0 * BALL_RADIUS_MM ** 2 / (5.0 * ball.speed * TIP_MM)
    return float(np.clip(tips, -MAX_TIPS, MAX_TIPS))


# What a rail does to the ball that hits it. The angle it comes off at with no
# side on it is measured - 1297 bounces off these tables - and kept as it was;
# what spin adds is put on top, because the measured curve already contains the
# widening a rolling ball gets and no model of side will produce that.
RAIL_FRICTION = 0.18
# How much side survives the compression. Swept against 20 tracked plays, with
# the tip position fitted per play: 0.55 lands the cue ball 93 mm from where the
# camera saw it a second in, against 105 mm at 0.75.
RAIL_KEEPS_SIDE = 0.55
# A cushion meets the ball above its equator, so it leaves mostly rolling
# rather than sliding. How much slide is left decides how far the ball then
# goes, and it is the one number here fitted rather than measured: at 0.3 a cue
# ball opening at the professionals' median 2699 mm/s runs 5821 mm against the
# 5680 mm the video records. Leaving it sliding gives 4093 - a table that eats
# shots - and leaving it purely rolling gives 7016.
RAIL_KEEPS_SLIDE = 0.3
REBOUND_IN = np.array([0.0, 6.7, 17.8, 27.2, 38.0, 46.4, 56.0, 66.1, 79.3, 90.0])
REBOUND_OUT = np.array([0.0, 16.7, 33.7, 41.8, 50.1, 55.4, 63.0, 70.6, 80.1, 90.0])
REBOUND_SPEED = np.array([0.817, 0.817, 0.844, 0.829, 0.824, 0.818, 0.808, 0.835, 0.912, 0.912])

RAILS = {"left": (1.0, 0.0), "right": (-1.0, 0.0),
         "top": (0.0, 1.0), "bottom": (0.0, -1.0)}


def bounce(ball, rail):
    """Send a ball back off a rail, with the side it was carrying.

    The tangential part is where side lives: the point of the ball touching the
    rail is moving at the ball's own speed along it plus what the spin adds, and
    friction works on the difference. Running side is the case where spin and
    travel agree, so friction has less to take away and the ball comes off wider
    and faster along the rail; reverse side is the opposite, and it is why a
    ball can be made to come off a cushion almost square.

    The ball keeps part of its side and loses the rest to the compression.
    """
    normal = np.asarray(RAILS[rail], dtype=float)
    tangent = np.array([-normal[1], normal[0]])

    into = float(np.dot(ball.velocity, normal))
    along = float(np.dot(ball.velocity, tangent))
    if into >= 0:
        return  # already leaving

    # The measured rebound, with no side on the ball.
    speed = float(np.hypot(into, along))
    incoming = np.degrees(np.arctan2(abs(along), abs(into)))
    outgoing = float(np.interp(incoming, REBOUND_IN, REBOUND_OUT))
    kept = float(np.interp(incoming, REBOUND_IN, REBOUND_SPEED)) * speed
    out_normal = kept * np.cos(np.radians(outgoing))
    out_along = np.sign(along) * kept * np.sin(np.radians(outgoing)) if along else 0.0

    # What the side does on top of it. The point of the ball touching the rail
    # is at -R along the normal, so the spin carries it *against* the ball's own
    # travel: the slip along the rail is v - Rω, not v + Rω. Getting that sign
    # wrong makes a cushion add spin instead of eating it, and running side then
    # closes the angle instead of opening it.
    slip = out_along - BALL_RADIUS_MM * ball.side
    grip = RAIL_FRICTION * (1.0 + 0.7) * abs(into)
    change = -np.sign(slip) * min(grip, (2.0 / 7.0) * abs(slip))
    out_along += change

    ball.velocity = normal * out_normal + tangent * out_along
    ball.side = (ball.side - (5.0 / (2.0 * BALL_RADIUS_MM)) * change) * RAIL_KEEPS_SIDE
    ball.slip = ball.velocity * RAIL_KEEPS_SLIDE


# Two balls do not part exactly along the line joining their centres. The one
# being struck is thrown a little to the side, by the friction between them, and
# how far depends on how the surfaces are moving past each other at the moment
# they touch - which is the cut angle and the side on the cue ball.
BALL_FRICTION = 0.06
BALL_RESTITUTION = 0.944
CUE_CARRY = 0.089


def collide(striker, struck_ball):
    """Resolve a contact between two balls, with throw and carry."""
    line = struck_ball.position - striker.position
    gap = float(np.hypot(*line))
    if gap == 0:
        return
    normal = line / gap
    tangent = np.array([-normal[1], normal[0]])

    approach = float(np.dot(striker.velocity - struck_ball.velocity, normal))
    if approach <= 0:
        return

    # Along the centres: the usual exchange, with a little of the striker's roll
    # carrying it forward through the contact.
    share = (1.0 + BALL_RESTITUTION) / 2.0
    transfer = share * approach
    striker_normal = float(np.dot(striker.velocity, normal)) - transfer \
        + CUE_CARRY * abs(approach)
    struck_normal = float(np.dot(struck_ball.velocity, normal)) + transfer

    # Across them: the surfaces rub. Cut angle and side both show up here.
    striker_tangent = float(np.dot(striker.velocity, tangent))
    surface = striker_tangent + BALL_RADIUS_MM * striker.side
    throw = -np.sign(surface) * min(BALL_FRICTION * transfer, abs(surface) * 0.5)
    struck_tangent = float(np.dot(struck_ball.velocity, tangent)) - throw

    striker.velocity = normal * striker_normal + tangent * (striker_tangent + throw)
    struck_ball.velocity = normal * struck_normal + tangent * struck_tangent
    # The striker keeps most of its side through a contact; the struck ball
    # picks a little up, spinning the other way.
    passed = throw * 5.0 / (2.0 * BALL_RADIUS_MM)
    struck_ball.side += passed
    striker.side = striker.side * 0.9 - passed * 0.2
    striker.slip = striker.velocity.copy()
    struck_ball.slip = struck_ball.velocity.copy()
