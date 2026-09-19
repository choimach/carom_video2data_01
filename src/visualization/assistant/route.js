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
      if (between.length >= LONG_AROUND_CUSHIONS) return "대회전";

      // 떠나온 쿠션으로 되돌아온다: 장-단-장이 같은 장쿠션이거나, 단-장-단이
      // 같은 단쿠션이면 되돌아오기. 선수의 정의 그대로다.
      if (between.length >= 3 && between[0] === between[2]
          && (SHORT.has(between[1]) !== SHORT.has(between[0]))) return "되돌아오기";

      // 단쿠션을 거치지 않고 두 장쿠션 사이를 건너다니면 횡단이다.
      if (between.length >= 3 && between.slice(0, 3).every((r) => LONG.has(r))
          && between[0] !== between[1] && between[1] !== between[2]) return "횡단";
    }

    // 쿠션이 셋에 못 미쳐도 계열은 읽힌다 — 어느 면을 맞고 어느 쪽으로 도는지가
    // 첫 쿠션에서 이미 정해지기 때문이다. 파이썬도 그렇게 한다. 조언판과 열거기는
    // 득점한 샷만 넘기므로 여기 오는 것은 언제나 셋 이상이지만, 두 쪽이 같은
    // 답을 내야 시험이 의미가 있다.
    // 계열을 가르는 유일한 기준: 맞힌 면과 **테이블을 도는 방향**이 같은 손이냐.
    // "오른쪽으로 돈다"는 접촉 순간의 꺾임이 아니다 — 오른쪽 면을 맞으면 수구는
    // 당장은 왼쪽으로 꺾인다. 그 다음 첫 쿠션이 단쿠션이냐 장쿠션이냐가 둘씩
    // 가른다.
    const sameHand = (face === "right") === circuitRight;
    const shortFirst = SHORT.has(between[0]);
    if (sameHand) return shortFirst ? "앞돌리기" : "옆돌리기";
    return shortFirst ? "빗겨치기" : "뒤돌리기";
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
    return name({
      before, between, reachedSecond: !!second, face,
      circuitRight: circuitIsRight(shot.paths[cue] || [], length, width),
    });
  }

  return { name, of, circuitIsRight, SHORT, LONG, LONG_AROUND_CUSHIONS };
})();

if (typeof module !== "undefined") module.exports = ROUTE;
