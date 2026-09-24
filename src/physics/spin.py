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
# 이 셋은 영상에 맞춰 잡았다. tools/fit_spin.py로 150개 플레이를 다시 쳐서
# 1초·2초 뒤 위치, 쿠션 개수, 이동거리 넷을 한꺼번에 본다. 교과서 값
# (미끄럼 0.20, 구름 0.0106)은 풀 당구천 기준이고, 캐롬 대대는 열선이 들어간
# 시모니스라 더 빠르다 — 교과서 값으로는 시뮬레이터 공이 실제의 72%밖에 못
# 가고 쿠션을 하나 덜 먹었다. 아래 값으로 95%, 쿠션 개수 차이는 중앙값 0,
# 1초 뒤 위치는 340 mm에서 253 mm로 줄었다.
SLIDING_FRICTION = 0.12
ROLLING_FRICTION = 0.006
# Spin about the vertical axis dies on its own, slowly: it has only the contact
# patch to work against. Slowly enough that over the first second - as far as
# the paths here are compared - the value makes no difference at all: 0.01,
# 0.022 and 0.04 give the same answer to the millimetre. It is left at the low
# end, and whatever settles it will have to be a shot measured further out.
# 0.01 / 0.03 / 0.08을 같은 150개 플레이로 재봤다: 1초 뒤 위치는 셋이 거의
# 같지만(283 / 281 / 291 mm) 쿠션 개수와 이동거리는 값을 올릴수록 나빠진다
# (차이 0 → -0.5 → -1.0, 비율 0.91 → 0.88 → 0.86). 데이터는 회전이 천 위에서
# 천천히 죽는 쪽을 고른다. 쿠션이 회전의 60% 이상을 먹으므로(RAIL_KEEPS_SIDE),
# 3쿠션 뒤 남는 회전은 어차피 쿠션이 정한다.
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
        # 마지막으로 맞은 쿠션에서 회전이 정이었나 역이었나. 리버스를 가리는 데
        # 쓴다 - 쿠션 차례로는 안 갈리는 유일한 유형이다.
        self.last_english = None
        self.last_spin_ratio = 0.0
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
    # 부호: 이 좌표계는 화면과 같아서 y가 아래로 간다. 그래서 "위에서 보아
    # 시계 방향"인 오른쪽 회전이 ω로는 음수다. 종이에 그려 따지면 손잡이를
    # 틀리기 쉬워서, tools/check_side_sign.py로 영상에 대고 판정했다: 회전
    # 방향이 측정된 플레이 120개에서 이 부호가 1초 뒤 301 mm, 반대 부호는
    # 364 mm였다 — 회전을 아예 안 넣은 348 mm보다도 나빴다. 선수가 먼저
    # 알아챘다: "당점결정을 좌우가 바뀌는 것 같아."
    side = -5.0 * speed * sideways / (2.0 * BALL_RADIUS_MM ** 2)

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
    tips = -ball.side * 2.0 * BALL_RADIUS_MM ** 2 / (5.0 * ball.speed * TIP_MM)
    return float(np.clip(tips, -MAX_TIPS, MAX_TIPS))


# What a rail does to the ball that hits it. The angle it comes off at with no
# side on it is measured - 1297 bounces off these tables - and kept as it was;
# what spin adds is put on top, because the measured curve already contains the
# widening a rolling ball gets and no model of side will produce that.
# 2026-09-24에 영상에 대고 다시 맞췄다. 프로의 실제 겨냥·강도로 다시 치고,
# **첫 쿠션 자리로 당점을 맞춘 뒤 둘째 쿠션이 얼마나 빗나가는지** 잰다
# (`tools/check_after_cushion.js`). 고르기가 빠지므로 물리만 남는다.
#
#   쿠션마찰 · 미끄럼남김    2쿠션    3쿠션   2팁 회전이득
#   0.18 · 0.0 (옛것)      261mm   389mm     13.2도
#   0.08 · 0.3 (지금)      198mm   233mm      9.8도
#   영상이 말하는 값                          10.8도
#
# 안 본 경기로 잰 값이다 (경기 단위로 반씩 갈랐다). 두 자가 서로 반대로 당기는
# 구간이 있어 — 마찰을 더 낮추면 쿠션 자리는 더 맞지만 회전 이득이 5.8도까지
# 떨어진다 — 둘 다 받아들일 만한 자리를 골랐다.
#
# 미끄럼 남김은 1.0이 쿠션 자리에는 가장 좋았지만(175mm) 분리각 시험을 깬다 —
# 1.0은 "공이 쿠션을 완전히 미끄러지며 떠난다"는 뜻이라 물리적으로도 과하다.
# 0.3이면 3쿠션 이득을 거의 다 가져오면서(233mm) 시험이 통과한다.
#
# 왜 0.08이 터무니없지 않은가: 캐롬 테이블은 전기로 데운다 (실온보다 약 5도,
# 국제 대회 규정상 필수). 천의 습기를 빼서 테이블을 빠르게 만든다. 학습에 쓰는
# 영상은 전부 대회 영상이므로 전부 가열 테이블이고, **선수가 실제로 치는
# 당구장 테이블도 가열된다**(본인 확인, 2026-09-25). 같은 세계라서 여기서 맞춘
# 값이 그의 실전에 그대로 간다.
#
# ⚠️ 셋째 자(사람다운 오차에서의 득점률)는 물리를 바꾸는 비교에 **쓸 수 없다**.
# 후보 줄들이 옛 물리로 찾은 것이라, 물리가 바뀌면 그 줄이 아예 득점하지 않아
# 무조건 무너진다. 탐색을 다시 돌린 뒤에야 쓸 수 있다.
RAIL_FRICTION = 0.08
# How much side survives the compression. Swept against 20 tracked plays, with
# the tip position fitted per play: 0.55 lands the cue ball 93 mm from where the
# camera saw it a second in, against 105 mm at 0.75.
RAIL_KEEPS_SIDE = 0.55
# 쿠션 아래로는 공이 기대고 있는 것이지 튕긴 것이 아니다. 이 선이 없으면 구석에
# 갇힌 공이 한 샷에 쿠션을 370개 먹고, judge()가 쿠션을 세므로 3쿠션을 넘긴 적
# 없는 샷이 득점으로 둔갑한다.
CUSHION_SPEED_MM_S = 60.0
# 쿠션에서 "회전을 적게 준" 것으로 칠 선. 회전이 만드는 표면 속도가 쿠션을
# 따라가는 속도의 이 비율보다 작으면 없다시피로 본다. ⚠️ 재서 얻은 값이 아니라
# 내가 고른 값이다.
LITTLE_SPIN = 0.15
# A cushion meets the ball above its equator, so it leaves rolling. The 0.3 that
# stood here was fitted against a travel figure the camera had truncated - 381
# of 426 tracked balls were still moving when the recording window closed - and
# it made the table eat shots. Refitted against position, cushion count and
# travel together, every one of them prefers zero: the ball leaves the rail
# rolling, which is also what the geometry says.
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

    Returns True when it actually reflected. A ball creeping along a rail comes
    back here every step, and answering "already leaving" without saying so let
    the caller record a cushion each time - one shot came back with 370 of them,
    and `judge` counts cushions, so shots that never made three were being
    called points.
    """
    normal = np.asarray(RAILS[rail], dtype=float)
    tangent = np.array([-normal[1], normal[0]])

    into = float(np.dot(ball.velocity, normal))
    along = float(np.dot(ball.velocity, tangent))
    if into >= 0:
        return False  # already leaving
    counts = ball.speed >= CUSHION_SPEED_MM_S
    # 들어갈 때 이 회전이 진행을 돕고 있었나(정회전) 거스르고 있었나(역회전).
    # 쿠션을 따라가는 방향 `along`과 회전이 만드는 표면 속도 Rω의 부호가 같으면
    # 미끄럼(v − Rω)이 줄어드니 정회전이고, 다르면 역회전이다. 리버스는 이것이
    # 1쿠션에서 역, 2쿠션에서 정인 샷이다.
    # 부호만으로는 "회전을 적게 준" 경우를 잡을 수 없다 - 아주 작은 순회전도
    # running이 된다. 그래서 크기까지 본다: 회전이 만드는 표면 속도 Rω가 쿠션을
    # 따라가는 속도의 몇 분의 몇이냐.
    reach = abs(along)
    ratio = (BALL_RADIUS_MM * ball.side * np.sign(along) / reach) if reach > 1e-6 else 0.0
    ball.last_spin_ratio = float(ratio)
    # 0.15는 "없다시피"의 선이고, 내가 정한 값이다 - 재서 얻은 것이 아니다.
    ball.last_english = ("none" if abs(ratio) < LITTLE_SPIN
                         else ("running" if ratio > 0 else "reverse"))

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
    return counts


# Two balls do not part exactly along the line joining their centres. The one
# being struck is thrown a little to the side, by the friction between them, and
# how far depends on how the surfaces are moving past each other at the moment
# they touch - which is the cut angle and the side on the cue ball.
BALL_FRICTION = 0.06
BALL_RESTITUTION = 0.944
# 수구가 충돌 뒤에도 앞으로 나아가는 것은 **남은 구름**이 하는 일이지 상수가
# 하는 일이 아니다. 여기 0.089가 들어 있던 동안에는 그것이 밀어치기를 흉내
# 내고 있었고, 그래서 스턴으로 쳐도 분리각이 90도가 아니라 74도였다. 회전이
# 충돌을 통과하게 고친 지금은 이중 계산이라 0으로 둔다 - 법선 방향으로 남는
# 것은 반발계수가 정하는 (1-e)/2 = 2.8%뿐이다.
CUE_CARRY = 0.0


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

    was_rolling = striker.velocity - striker.slip
    striker.velocity = normal * striker_normal + tangent * (striker_tangent + throw)
    struck_ball.velocity = normal * struck_normal + tangent * struck_tangent
    # The striker keeps most of its side through a contact; the struck ball
    # picks a little up, spinning the other way.
    passed = throw * 5.0 / (2.0 * BALL_RADIUS_MM)
    struck_ball.side += passed
    striker.side = striker.side * 0.9 - passed * 0.2

    # 분리각은 여기서 나온다. 두 공 사이의 충격은 중심을 잇는 선을 따라 두
    # 중심을 지나므로 **수평축 회전에는 토크를 주지 않는다** - 수구가 갖고 있던
    # 구름은 충돌을 그대로 통과한다. 속도만 바뀌고 회전은 남으므로, 접촉점의
    # 속도는 새 속도에서 그 남은 구름을 뺀 것이다.
    #
    # 여기를 `slip = velocity`로 두면 "수구는 늘 미끄러지며 떠난다"고 말하는
    # 셈이고, 그러면 분리각이 당점에도 세기에도 거리에도 꿈쩍하지 않는다 -
    # 실제로 그랬다. 선수의 말: "분리각은 스트로크가 강할수록, 당점이 아래쪽
    # 으로 갈수록 커지고 vice versa." 그 셋은 모두 접촉 순간의 미끄럼 하나로
    # 설명되는데, 그 미끄럼을 충돌이 지우고 있었다.
    striker.slip = striker.velocity - was_rolling
    # 맞은 공은 서 있었으므로 회전이 없다. 온전히 미끄러지며 떠난다.
    struck_ball.slip = struck_ball.velocity.copy()
