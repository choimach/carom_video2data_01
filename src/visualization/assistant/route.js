// 경로에 이름 붙이기 — 한 벌.
//
// 이 판정이 한때 **세 벌**로 갈라져 있었다. 조언판에는 되돌아오기가 있고
// 대회전·횡단이 없었고, tools/enumerate_alternatives.js에는 대회전만 있었고,
// src/physics/route.py에는 셋 다 있었다. 그래서 프로가 되돌아오기로 친 12판을
// 열거기가 한 판도 재현하지 못했다 — 탐색이 못 찾은 것이 아니라 **그 이름을
// 만들 줄 몰랐던** 것이다.
//
// 그러므로 여기 한 벌만 둔다. 파이썬과 같아야 하고 tests/test_route_js.py가
// 둘을 대조한다. 판정의 근거는 전부 ref/taxonomy.md §5에 있다 — 선수가 말로
// 준 정의이고, 내가 라벨을 보고 추측했던 규칙 셋은 전부 틀렸다.

const ROUTE = (() => {
  const SHORT = new Set(["left", "right"]);     // 단쿠션 (2844 mm 축의 양 끝)
  const LONG = new Set(["top", "bottom"]);      // 장쿠션
  const LONG_AROUND_CUSHIONS = 5;

  // before        : 1적구보다 먼저 맞은 쿠션 수
  // between       : 1적구와 2적구 사이에 맞은 쿠션 이름들, 순서대로
  // reachedSecond : 2적구에 닿았는가
  // face          : 1적구의 어느 면을 맞았나 ("left" | "right")
  // circuitRight  : 테이블을 오른쪽으로 돌았나 (경로가 쓸고 간 부호 있는 넓이)
  function name({ before, between, reachedSecond, face, circuitRight }) {
    // 1적구보다 먼저 쿠션을 맞으면 그건 "도는" 것이 아니다.
    if (before >= 2) return "뱅크샷";
    if (before === 1) return "걸어치기";
    if (!between || between.length === 0) return null;

    if (reachedSecond) {
      // 횡단과 더블은 둘 다 "마주보는 두 쿠션 사이를 오간다". 가르는 것은 몇 번
      // 오갔느냐다 (2026-09-20, 그가 말로 준 기준):
      //
      //   횡단  — 두 장쿠션 사이를 **3회 이상** 오간 뒤 득점
      //   더블  — **2회** 오간 뒤, 3쿠션째에 단쿠션을 맞고 득점
      //
      // "아주 드물게는 두 단쿠션 사이를 오가면서도 가능"하다고 해서, 장·단을
      // 가리지 않고 마주보는 한 쌍을 오가는 것으로 센다. 바깥 자료와도 맞는다:
      // "단쿠션에 나란하게 왕복하며 최종적으로는 앞으로 전진하여 득점"
      // (japong.com) — 단쿠션과 나란히 오가면 부딪히는 벽은 장쿠션이다.
      const crossing = crossingRun(between);
      if (crossing >= 3) return "횡단";
      // 더블은 **최상위 이름이 아니다**. 2026-09-20에 확인: "더블은 빗겨치기의
      // 하위구분이 맞아." 그래서 여기서 이름을 가로채지 않고 계열 규칙에
      // 맡긴다. 하위 구분은 subtypeOf()가 따로 돌려준다.

      // ★대회전은 **최상위 이름이 아니다** — 더블과 같다. 선수가 확정해 줬다
      // (2026-09-26): *"그래서 대회전에는 유형이 같이붙어 — 옆돌리기 대회전,
      // 뒤돌리기 대회전 등."* `ref/taxonomy.md`의 예시도 그렇다.
      //
      // 옛 코드는 여기서 `if (between.length >= 5) return "대회전";`으로 **계열
      // 규칙에 닿기 전에 이름을 가로챘다.** 그래서 바탕이 뒤돌리기였는지
      // 옆돌리기였는지가 사라지고, "뒤돌리기라면 1쿠션은 장쿠션" 같은 조건도
      // 같이 사라졌다. 선수가 화면을 보고 짚은 것이 그것이다 (씨앗 596883888):
      // *"이 그림은 대회전 뒤돌리기인데 1쿠션이 장쿠션이 아닌 단쿠션을 보여주고
      // 있어, 당점이 반대방향이야."* 1쿠션이 틀리면 거기 필요한 회전도 반대다.
      //
      // 덤: `tools/score_probability.py`가 대회전을 빼야 했던 오염도 풀린다 —
      // 2적구 전 쿠션 수로 이름을 붙이면 **실패한 판이 그 이름을 가질 수 없어**
      // 자료에 100% 득점으로 잡힌다. 바탕 이름은 실패한 판도 담는다.

      // 떠나온 쿠션으로 되돌아온다: 장-단-장이 같은 장쿠션이거나, 단-장-단이
      // 같은 단쿠션이면 되돌아오기. 그가 말로 준 것을 그대로 옮겼다 (2026-09-19).
      // 그는 프로가 아니라 배우는 사람이고 본인도 100% 확신하지 않는다 —
      // 라벨 30개에 대고 rules_check.py로 점수를 받는 가설로 다룬다.
      if (between.length >= 3 && between[0] === between[2]
          && (SHORT.has(between[1]) !== SHORT.has(between[0]))) return "되돌아오기";
    }

    // 쿠션이 셋에 못 미쳐도 계열은 읽힌다 — 어느 면을 맞고 어느 쪽으로 도는지가
    // 첫 쿠션에서 이미 정해지기 때문이다. 파이썬도 그렇게 한다. 조언판과 열거기는
    // 득점한 샷만 넘기므로 여기 오는 것은 언제나 셋 이상이지만, 두 쪽이 같은
    // 답을 내야 시험이 의미가 있다.
    // 계열을 가르는 유일한 기준: 맞힌 면과 **테이블을 도는 방향**이 같은 손이냐.
    // "오른쪽으로 돈다"는 접촉 순간의 꺾임이 아니라 경로 전체가 테이블을 도는
    // 방향이다. 꺾임 자체는 **맞힌 면 쪽으로** 일어난다 (선수, 2026-09-21:
    // "반사각이 움직이는 쪽이 맞는 면 쪽이야" — 시뮬레이터로 확인했다).
    // 그 다음 첫 쿠션이 단쿠션이냐 장쿠션이냐가 둘씩 가른다.
    const sameHand = (face === "right") === circuitRight;
    const shortFirst = SHORT.has(between[0]);
    if (sameHand) return shortFirst ? "앞돌리기" : "옆돌리기";
    return shortFirst ? "빗겨치기" : "뒤돌리기";
  }

  // 하위 구분 — 최상위 이름과 나란히 붙는 꼬리표이지 이름 자체가 아니다.
  //
  // ⚠️ 여기에 풀리지 않은 것이 있다. 그가 말한 더블쿠션은 **장-장-단**인데
  // (두 장쿠션 사이를 두 번 오간 뒤 단쿠션), ref/taxonomy.md의 빗겨치기 행은
  // 패턴이 **단-단-장**이다. 계열 규칙으로 장쿠션이 먼저면 빗겨치기가 아니라
  // 뒤돌리기 쪽이다. 둘 중 무엇이 더블인지 아직 정해지지 않았다.
  function subtypeOf(between, english, reachedSecond) {
    const tags = [];
    if (between && between.length >= 3 && crossingRun(between) === 2
        && SHORT.has(between[2]) !== SHORT.has(between[0])) tags.push("더블");

    // 대회전 — 2적구 전에 쿠션을 다섯 번 넘게 맞으면 테이블을 크게 돈 것이다.
    // 이름이 아니라 꼬리표다 (위 name()의 주석 참고).
    //
    // ⚠️ **2적구에 닿은 샷에만** 붙인다. 빗나간 뒤 계속 구른 쿠션은 "무언가로
    // 가는 길"이 아니다 — 옛 이름 규칙이 지키던 조건이고 꼬리표에도 그대로다.
    if (reachedSecond && between && between.length >= LONG_AROUND_CUSHIONS) {
      tags.push("대회전");
    }

    // 리버스 — "역회전으로 1쿠션을 맞히고 두 번째 쿠션부터는 제회전으로 진행"
    // (japong.com). 쿠션 차례로는 갈리지 않고 **회전으로만** 갈리는 유일한
    // 것이라, sim.js가 쿠션마다 남긴 정/역을 읽는다.
    //
    // ⚠️ 꼬리표로 둔 것은 확신이 낮은 결정이다. 그의 말 그대로 (2026-09-20):
    // "리버스는 꼬리표가 맞아 — 글쎄... 그렇게 두는 게 편하긴 하겠지... 나중에
    // 프로그램이 더 진화하면 더 정확히 하고..." 편하다는 것이 이유의 일부다.
    if (english && english.length >= 2
        && english[0] === "reverse" && english[1] === "running") tags.push("리버스");
    return tags.length ? tags : null;
  }

  // 세워치기 — 앞돌리기인데 회전을 적게 주거나 역으로 주어 **반사각을 좁힌** 것.
  // 그가 준 정의 (2026-09-20): "세워치기는 앞돌리기의 하위구분이고 회전을 적게
  // 주거나 어느정도의 역회전을 주어서 반사각이 적게 만들어서 공이 길게 들어오게
  // 만드는 방법."
  //
  // 바깥 자료에는 가르는 정의가 없었고 영어 이름조차 Long inside angle shot과
  // Short angle shot으로 엇갈렸다. 저장소의 ref/carom_technic.txt는 "큐를 세워
  // 치는 타법"이라 적고 있었는데, 그의 말은 큐가 아니라 회전과 반사각이다.
  function standing(route, english) {
    return route === "앞돌리기" && !!english
      && (english[0] === "reverse" || english[0] === "none");
  }

  // 마주보는 두 쿠션을 번갈아 맞은 횟수 — 처음부터 이어지는 만큼만 센다.
  // src/physics/route.py의 _crossing_run과 같은 식이어야 한다.
  function crossingRun(rails) {
    if (!rails.length) return 0;
    const sameKind = SHORT.has(rails[0]);
    let run = 1;
    for (let i = 1; i < rails.length; i++) {
      if (rails[i] === rails[i - 1] || SHORT.has(rails[i]) !== sameKind) break;
      run += 1;
    }
    return run;
  }

  // 경로가 테이블 한가운데를 기준으로 쓸고 간 부호 있는 넓이. 한 순간의 꺾임이
  // 아니라 경로 전체의 도는 방향이다.
  function circuitIsRight(path, length, width) {
    let area = 0;
    for (let i = 0; i + 1 < path.length; i++) {
      const ax = path[i][0] - length / 2, ay = path[i][1] - width / 2;
      const bx = path[i + 1][0] - length / 2, by = path[i + 1][1] - width / 2;
      area += ax * by - ay * bx;
    }
    return area > 0;
  }

  // 한 샷(sim.js의 결과)에서 위 함수가 필요한 것들을 뽑아 이름을 돌려준다.
  function of(shot, judged, face, cue, length, width) {
    const balls = shot.events.filter((e) => e.kind === "ball");
    if (!balls.length) return null;
    const first = balls[0];
    const second = balls.find((e) => e.detail !== first.detail);
    const before = shot.events
      .filter((e) => e.kind === "cushion" && e.at < first.at).length;
    const between = shot.events
      .filter((e) => e.kind === "cushion" && e.at > first.at
        && (!second || e.at < second.at))
      .map((e) => e.detail);
    const english = shot.events
      .filter((e) => e.kind === "cushion" && e.at > first.at
        && (!second || e.at < second.at))
      .map((e) => e.english);
    const route = name({
      before, between, reachedSecond: !!second, face,
      circuitRight: circuitIsRight(shot.paths[cue] || [], length, width),
    });
    const tags = subtypeOf(between, english, !!second) || [];
    if (standing(route, english)) tags.push("세워치기");
    return { route, tags: tags.length ? tags : null };
  }

  return { name, of, subtypeOf, standing, circuitIsRight, crossingRun, SHORT, LONG, LONG_AROUND_CUSHIONS };
})();

if (typeof module !== "undefined") module.exports = ROUTE;
