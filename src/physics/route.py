"""What kind of shot was this, in the words a player uses.

The model's answer has to be a route a player recognises - 뒤돌리기, 옆돌리기,
앞돌리기 - not "three cushions, first one long". `ref/taxonomy.md` holds the
vocabulary; this turns a play's contact sequence into one of its names.

Some of the splits are solid and some are guesses, and the difference matters
more than the code does:

* Solid, because they are counts of things the pipeline already detects. How
  many cushions the cue ball took before the first object ball (none, one,
  more) separates a direct shot from 걸어치기 and 뱅크샷. How many it took
  before the second object ball separates 대회전 from the rest.
* A hypothesis, because it is fitted to ten plays the player named himself:
  which of 뒤돌리기, 옆돌리기, 앞돌리기 and 빗겨치기 a direct shot is. See
  `_turn` for what those ten said - and for why the rule that preceded it, over
  rail sequences, could never have worked.

So every verdict carries `basis`: "counted" where it rests on the first kind of
split, "geometry" where it rests on the second. The second is worth no more
than the ten labels under it until a wider batch has been checked.
"""

# Rails are named for where they sit: the table is 2844 x 1422, x along the
# length, so "left" and "right" are the short rails a player calls 단쿠션 and
# "top" and "bottom" the long ones, 장쿠션.
SHORT_RAILS = ("left", "right")
LONG_RAILS = ("top", "bottom")

DIRECT = "직접"
HOOK = "걸어치기"
BANK = "뱅크샷"
LONG_AROUND = "대회전"
CROSSING = "횡단"
BEHIND = "뒤돌리기"
SIDE = "옆돌리기"
FRONT = "앞돌리기"
GLANCING = "빗겨치기"
UNKNOWN = "미분류"

LONG_AROUND_CUSHIONS = 5
# Both measured off the player's own labels, on two examples each: the one
# 빗겨치기 he named was struck at 0.06 and nothing else came under 0.21, and the
# two 옆돌리기 turned 78 and 96 degrees where the two 뒤돌리기 turned none at all.
THIN_HIT = 0.15
SIDE_TURN_DEGREES = 60.0


def _rails(events, after=None, before=None):
    return [e.detail for e in events
            if e.kind == "cushion"
            and (after is None or e.frame > after)
            and (before is None or e.frame < before)]


def classify(events, layout_mm=None, cue_ball=None, thickness=None, turn_deg=None):
    """Name the route this shot took.

    `events` is what `judge_shot` leaves in its verdict: the cue ball's
    contacts, in order, each a cushion by rail name or a ball by colour.
    `thickness` and `turn_deg` are what separate the turns from one another;
    without the turn angle, 뒤돌리기 and 옆돌리기 cannot be told apart and come
    back as 미분류. `layout_mm` and `cue_ball` are no longer read here.

    Returns {route, basis, opening, cushions, rails} - `opening` being the
    cushions taken before the first object ball, which is what makes a shot
    걸어치기 or 뱅크샷 rather than a turn.
    """
    balls = [e for e in events if e.kind == "ball"]
    if not balls:
        return {"route": UNKNOWN, "basis": "counted", "opening": 0,
                "cushions": 0, "rails": [], "why": "no object ball was touched"}

    first = balls[0]
    second = next((e for e in balls if e.detail != first.detail), None)
    opening = _rails(events, before=first.frame)
    between = _rails(events, after=first.frame,
                     before=second.frame if second else None)

    result = {"opening": len(opening), "cushions": len(between),
              "rails": between, "basis": "counted",
              "reached_second": second is not None}

    # A cushion before the object ball is not a turn at all: one is 걸어치기,
    # where the rail is used to reach a ball that cannot be hit straight; more
    # than one and the cue ball is being banked around, which the notes file
    # under 구멍.
    if len(opening) >= 2:
        return {**result, "route": BANK, "why": f"{len(opening)} cushions first"}
    if len(opening) == 1:
        return {**result, "route": HOOK, "why": "a cushion before the first ball"}

    if not between:
        return {**result, "route": UNKNOWN, "basis": "counted",
                "why": "no cushion between the object balls"}

    # A shot that never reached a second object ball has no "cushions before
    # the second ball" to count: what gets counted instead is every rail the
    # cue ball took until it stopped, which is why 대회전 came out at one play
    # in eight. The route it set off on is still readable from the first rail,
    # so the turn is named and the counted routes are left alone.
    reached_second = result["reached_second"]

    if reached_second:
        if len(between) >= LONG_AROUND_CUSHIONS:
            return {**result, "route": LONG_AROUND,
                    "why": f"{len(between)} cushions before the second ball"}

        # Crossing the table between the two long rails, without a short rail
        # in between, is 횡단 whatever else it resembles.
        if len(between) >= 3 and all(rail in LONG_RAILS for rail in between[:3]) \
                and between[0] != between[1] and between[1] != between[2]:
            return {**result, "route": CROSSING, "why": "long rail to long rail"}

    return {**result, **_turn(between, thickness, turn_deg)}


def _turn(between, thickness, turn_deg):
    """뒤돌리기, 옆돌리기, 앞돌리기 or 빗겨치기, from the player's ten.

    The first rule here read the rail sequence and was wrong seven times in
    ten. It was wrong in kind, not in degree: six of those plays took the same
    장-단-장 and the player called them by five different names, so no rule
    over rail sequences can name a route at all.

    What did line up with his names:

      두께 0.06        빗겨치기      - thin, and nothing else was near it
      단쿠션 먼저       앞돌리기      - the old rule called exactly this 뒤돌리기
      장쿠션, 꺾임 0°   뒤돌리기
      장쿠션, 꺾임 80°+ 옆돌리기

    Two examples each. That is a hypothesis with a sample of ten behind it, not
    a rule that has been checked, and 되돌아오기 - which he named twice - has no
    test here at all. The next batch of labels is what decides it.
    """
    if thickness is not None and thickness <= THIN_HIT:
        return {"route": GLANCING, "basis": "geometry",
                "why": f"struck thin, {thickness:.2f}"}
    if between[0] in SHORT_RAILS:
        return {"route": FRONT, "basis": "geometry",
                "why": "first rail is a short one"}
    if turn_deg is None:
        return {"route": UNKNOWN, "basis": "geometry",
                "why": "no turn angle to tell 뒤돌리기 from 옆돌리기"}
    if abs(turn_deg) >= SIDE_TURN_DEGREES:
        return {"route": SIDE, "basis": "geometry",
                "why": f"turned {abs(turn_deg):.0f}° off the object ball"}
    return {"route": BEHIND, "basis": "geometry",
            "why": f"long rail, barely turned ({abs(turn_deg):.0f}°)"}
