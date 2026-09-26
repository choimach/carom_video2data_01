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
DOUBLE = "더블"
REVERSE = "리버스"
STANDING = "세워치기"
BEHIND = "뒤돌리기"
SIDE = "옆돌리기"
FRONT = "앞돌리기"
GLANCING = "빗겨치기"
RETURNING = "되돌아오기"
UNKNOWN = "미분류"

LONG_AROUND_CUSHIONS = 5
# The one 빗겨치기 the player named was struck at 0.06 and nothing else he
# named came under 0.21.
THIN_HIT = 0.15
# Where 옆돌리기 ends and 뒤돌리기 begins, in millimetres of second cushion
# carried on along the line the cue ball was sent. Zero looked right until he
# named two shots at +208 and +214 as 옆돌리기; against 뒤돌리기 at +387, +460,
# +505 and +844, the line sits between them and not at nothing.
SIDE_DRIFT_MM = 300.0


def _rails(events, after=None, before=None):
    return [e.detail for e in events
            if e.kind == "cushion"
            and (after is None or e.frame > after)
            and (before is None or e.frame < before)]


def _crossing_run(rails):
    """마주보는 두 쿠션을 번갈아 맞은 횟수 — 처음부터 이어지는 만큼만 센다.

    ['top','bottom','top'] 이면 3, ['top','bottom','left'] 이면 2,
    ['top','left',...] 이면 1이다.
    """
    if not rails:
        return 0
    same_kind = rails[0] in SHORT_RAILS
    run = 1
    for before, rail in zip(rails, rails[1:]):
        if rail == before or (rail in SHORT_RAILS) != same_kind:
            break
        run += 1
    return run


def _subtype(rails, english=None, reached_second=False):
    """최상위 이름과 나란히 붙는 꼬리표들. 이름 자체는 아니다.

    ⚠️ 풀리지 않은 것: 그가 말한 더블쿠션은 **장-장-단**인데 (두 장쿠션 사이를
    두 번 오간 뒤 단쿠션), ref/taxonomy.md의 빗겨치기 행은 패턴이 **단-단-장**
    이다. 계열 규칙으로 장쿠션이 먼저면 빗겨치기가 아니라 뒤돌리기 쪽이다.
    둘 중 무엇이 더블인지 아직 정해지지 않았다.
    """
    tags = []
    if len(rails) >= 3 and _crossing_run(rails) == 2 \
            and (rails[2] in SHORT_RAILS) != (rails[0] in SHORT_RAILS):
        tags.append(DOUBLE)

    # 대회전 — 2적구 전에 쿠션을 다섯 번 넘게 맞으면 테이블을 크게 돈 것이다.
    # 이름이 아니라 꼬리표다 (위 name 쪽 주석 참고). 선수 (2026-09-26):
    # "대회전에는 유형이 같이붙어 — 옆돌리기 대회전, 뒤돌리기 대회전 등."
    # ⚠️ **2적구에 닿은 샷에만** 붙인다. 빗나간 뒤 계속 구른 쿠션은 "무언가로
    # 가는 길"이 아니다 — 옛 이름 규칙이 지키던 조건이고 꼬리표에도 그대로다.
    if reached_second and len(rails) >= LONG_AROUND_CUSHIONS:
        tags.append(LONG_AROUND)

    # 리버스 — "역회전으로 1쿠션을 맞히고 두 번째 쿠션부터는 제회전으로 진행".
    # 쿠션 차례로는 갈리지 않고 **회전으로만** 갈리는 유일한 것이라, 시뮬레이터가
    # 쿠션마다 남긴 정/역을 읽어야 한다. 영상 플레이에는 아직 이 값이 없다.
    if english and len(english) >= 2 \
            and english[0] == "reverse" and english[1] == "running":
        tags.append(REVERSE)
    return tags or None


def _standing(route, english):
    """세워치기 — 앞돌리기인데 회전을 적게 주거나 역으로 주어 반사각을 좁힌 것.

    그가 준 정의 (2026-09-20): *"세워치기는 앞돌리기의 하위구분이고 회전을 적게
    주거나 어느정도의 역회전을 주어서 반사각이 적게 만들어서 공이 길게 들어오게
    만드는 방법."*

    바깥 자료에는 가르는 정의가 없었다 - 영어 이름조차 Long inside angle shot과
    Short angle shot으로 엇갈렸다. 저장소의 `ref/carom_technic.txt`는 "큐를 세워
    치는 타법"이라고 적고 있었는데, 그의 말은 큐가 아니라 **회전과 반사각**에
    대한 것이다.
    """
    return (route == FRONT and english and english[0] in ("reverse", "none"))


def classify(events, layout_mm=None, cue_ball=None, thickness=None, turn_deg=None,
             away_mm=None, struck_side=None, english=None, circuit=None,
             english_at_rail=None):
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
        # 횡단과 더블은 둘 다 "마주보는 두 쿠션 사이를 오간다". 가르는 것은
        # 몇 번 오갔느냐다 (2026-09-20, 그가 말로 준 기준):
        #
        #   횡단  — 두 장쿠션 사이를 **3회 이상** 오간 뒤 득점
        #   더블  — **2회** 오간 뒤, 3쿠션째에 단쿠션을 맞고 득점
        #
        # "아주 드물게는 두 단쿠션 사이를 오가면서도 가능"하다고 해서, 장·단을
        # 가리지 않고 **마주보는 한 쌍**을 오가는 것으로 센다.
        #
        # 바깥 자료와도 맞는다: "단쿠션에 나란하게 왕복하며 최종적으로는 앞으로
        # 전진하여 득점" (japong.com). 단쿠션과 나란히 오가면 부딪히는 벽은
        # 장쿠션이다.
        crossing = _crossing_run(between)
        if crossing >= 3:
            return {**result, "route": CROSSING,
                    "why": f"{crossing} crossings between the same pair of rails"}
        # 더블은 **최상위 이름이 아니다**. 2026-09-20에 확인: "더블은 빗겨치기의
        # 하위구분이 맞아." 이름은 계열 규칙에 맡기고, 꼬리표만 붙인다.

        # 횡단을 대회전보다 먼저 본다: 그가 "3회 **이상**"이라고 했으므로 다섯 번
        # 오간 것도 횡단이지 대회전이 아니다. 대회전의 하위 구분은 뒤돌리기
        # 대회전·옆돌리기 대회전처럼 **테이블을 크게 도는** 것들이라, 오가는 것과
        # 기하가 다르다. ⚠️ 이 순서는 내 판단이고 확인받은 적이 없다.

        # ★대회전은 **최상위 이름이 아니다** — 더블과 같다. 선수가 확정해 줬다
        # (2026-09-26): "그래서 대회전에는 유형이 같이붙어 — 옆돌리기 대회전,
        # 뒤돌리기 대회전 등." ref/taxonomy.md의 예시도 그렇다.
        #
        # 옛 코드는 여기서 route를 LONG_AROUND로 바꿔 **계열 규칙에 닿기 전에
        # 이름을 가로챘다.** 그래서 바탕 계열이 사라지고, "뒤돌리기라면 1쿠션은
        # 장쿠션" 같은 조건도 같이 사라졌다. 선수가 화면에서 짚은 것이 그것이다.

        # Back to the rail it came off: 장-단-장 on one and the same long rail,
        # or 단-장-단 on one and the same short rail, is 되돌아오기 - the
        # player's definition, word for word.
        if len(between) >= 3 and between[0] == between[2] \
                and (between[1] in SHORT_RAILS) != (between[0] in SHORT_RAILS):
            return {**result, "route": RETURNING,
                    "why": f"back to the same {between[0]} rail"}


    named = {**result, **_turn(between, struck_side, circuit)}
    tags = _subtype(between, english_at_rail, bool(result.get("reached_second"))) or []
    if _standing(named.get("route"), english_at_rail):
        tags = [*tags, STANDING]
    return {**named, "tags": tags} if tags else named


def _turn(between, struck_side, circuit):
    """뒤돌리기, 옆돌리기, 앞돌리기 or 빗겨치기 - the player's own family rule.

    Two rules came before this one and both were mine, read off his labels
    rather than given by him: the rail sequence, which named the same 장-단-장
    five different things, and the second cushion's drift, which he then
    contradicted by naming shots at +214 and +537 옆돌리기 against 뒤돌리기 at
    +387 and +460.

    What he actually said, asked a third time: the cue ball strikes the right
    face of the object ball, carries right side, and goes round to the right -
    and that is 옆돌리기; opposite hands and it is 뒤돌리기. So the test is
    whether the face struck and the way the ball travels round the table agree.
    It is the 제각돌리기 / 빗겨치기 split the lesson sites describe, and against
    the 24 plays he has named by hand it gets 23.

    The rail that comes first then says which member of the family it is: a
    short rail means the near pair, 앞돌리기 and 빗겨치기.
    """
    if struck_side is None or circuit is None:
        return {"route": UNKNOWN, "basis": "geometry",
                "why": "no face or circuit to name the turn from"}
    # struck_side is positive for the object ball's left face; circuit is +1
    # going round to the right.
    same_hand = (struck_side < 0) == (circuit > 0)
    short_first = between[0] in SHORT_RAILS
    if same_hand:
        return {"route": FRONT if short_first else SIDE, "basis": "geometry",
                "why": "face struck and circuit on the same hand"}
    return {"route": GLANCING if short_first else BEHIND, "basis": "geometry",
            "why": "face struck and circuit on opposite hands"}
