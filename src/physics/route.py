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
* A guess, because it rests on geometry no one has checked: which of 뒤돌리기,
  옆돌리기 and 앞돌리기 a direct shot is. The rule below reads the first rail
  the cue ball reaches after the object ball - short rail means 뒤돌리기 - and
  then, for the long rails, which side of the table it stayed on. That matches
  how the notes describe these shots and nothing more. It has not been checked
  against a single play a person looked at.

So every verdict carries `basis`: "counted" where it rests on the first kind of
split, "geometry" where it rests on the second. Treat the second as unlabelled
data until someone has gone through a few dozen of them.
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
UNKNOWN = "미분류"

LONG_AROUND_CUSHIONS = 5


def _rails(events, after=None, before=None):
    return [e.detail for e in events
            if e.kind == "cushion"
            and (after is None or e.frame > after)
            and (before is None or e.frame < before)]


def classify(events, layout_mm=None, cue_ball=None):
    """Name the route this shot took.

    `events` is what `judge_shot` leaves in its verdict: the cue ball's
    contacts, in order, each a cushion by rail name or a ball by colour.
    `layout_mm` and `cue_ball` are only needed to tell 옆돌리기 from 앞돌리기;
    without them that pair comes back as 미분류.

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
              "rails": between, "basis": "counted"}

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
    if len(between) >= LONG_AROUND_CUSHIONS:
        return {**result, "route": LONG_AROUND,
                "why": f"{len(between)} cushions before the second ball"}

    # Crossing the table between the two long rails, without a short rail in
    # between, is 횡단 whatever else it resembles.
    if len(between) >= 3 and all(rail in LONG_RAILS for rail in between[:3]) \
            and between[0] != between[1] and between[1] != between[2]:
        return {**result, "route": CROSSING, "why": "long rail to long rail"}

    if between[0] in SHORT_RAILS:
        return {**result, "route": BEHIND, "basis": "geometry",
                "why": "first rail is a short one"}

    return {**result, **_long_rail_first(between, layout_mm, cue_ball)}


def _long_rail_first(between, layout_mm, cue_ball):
    """옆돌리기 or 앞돌리기 - the split that has not been checked.

    Both send the cue ball to a long rail off the object ball. What the notes
    distinguish is where it goes from there: 앞돌리기 curls in front of the
    object ball and stays on the cue ball's own side of the table, while
    옆돌리기 carries across it. So the test is whether the rail it reached is
    the one the cue ball was already nearer to.
    """
    if not layout_mm or not cue_ball:
        return {"route": UNKNOWN, "basis": "geometry",
                "why": "no layout to tell 옆돌리기 from 앞돌리기"}

    cue = layout_mm.get(cue_ball)
    others = [xy for colour, xy in layout_mm.items() if colour != cue_ball]
    if cue is None or len(others) < 2:
        return {"route": UNKNOWN, "basis": "geometry",
                "why": "the layout is incomplete"}

    # Which long rail the cue ball started nearer to. `top` is y = 0.
    from src.physics.table_calibration import TABLE_WIDTH_MM

    near_rail = "top" if cue[1] < TABLE_WIDTH_MM / 2 else "bottom"
    if between[0] == near_rail:
        return {"route": FRONT, "basis": "geometry",
                "why": "long rail on the cue ball's own side"}
    return {"route": SIDE, "basis": "geometry",
            "why": "long rail across the table"}
