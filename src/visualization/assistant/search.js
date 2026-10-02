// 탐색의 알맹이 — 한 줄을 쳐 보고 판정한다. **한 벌.**
//
// 2026-10-03까지 이것은 assistant.html 안에 있었고 화면 스레드에서 돌았다. 폰에서 12초가
// 걸렸고 그동안 화면이 멈췄다 (선수: "12초 걸림"). 시뮬레이터는 줄마다 완전히 독립이므로
// 워커 여럿에 나눈다. 그러려면 같은 코드를 **화면과 워커가 같이** 써야 한다 — 두 벌로
// 베끼면 조용히 어긋난다 (CLAUDE.md §5). 그래서 이 파일은:
//
//   · 화면에서는 <script src="search.js">로 읽혀 SEARCH를 내놓고 (워커가 안 될 때의 대비책),
//   · 워커에서는 new Worker("search.js")로 읽혀 sim.js · route.js를 스스로 불러 일감을 받는다.
//
// 배치·수구·조건은 전역에서 읽지 않고 ctx로 받는다:
//   { layout, cue, wantFace, wantFirst, gate, gateLadder, L, W }
//
// 돌려주는 hit에는 **shot이 없다** — 궤적은 수천 점이라 워커에서 화면으로 넘기기에 무겁다.
// 묶고 고르는 데 필요한 것(pathId, running, judged.first)만 싣고, 화면이 대표 줄만 다시 친다.
if (typeof importScripts === "function" && typeof document === "undefined") {
  importScripts("sim.js", "route.js");
}

const SEARCH = (() => {
  const ORDER = ["white", "yellow", "red"];
  const BIN_MM = 400;   // 첫 쿠션 자리 칸 — tools/enumerate_alternatives.js의 bucket()과 같다

  // 득점 못 한 줄이 2적구에 얼마나 가까이 갔나 (셋째 쿠션부터). 다듬을지 정하는 관문.
  function nearMiss(ctx, shot, first) {
    const { layout, cue } = ctx;
    const other = ORDER.find((c) => c !== cue && c !== first);
    if (!other || !layout[other]) return Infinity;
    const hitAt = (shot.events.find((e) => e.kind === "ball") || {}).at;
    if (hitAt === undefined) return Infinity;
    const after = shot.events.filter((e) => e.kind === "cushion");
    if (after.length < 3) return Infinity;
    const from = Math.max(Math.round(after[2].at * 60), Math.round(hitAt * 60));
    const path = shot.paths[cue] || [];
    const [x, y] = layout[other];
    let best = Infinity;
    for (let i = Math.min(from, path.length - 1); i < path.length; i++) {
      best = Math.min(best, Math.hypot(path[i][0] - x, path[i][1] - y));
    }
    return best;
  }

  // 키스 — 계산한 줄이 실제로는 성립하지 않는 두 가지.
  function kissed(shot) {
    const balls = shot.events.filter((e) => e.kind === "ball");
    if (!balls.length) return false;
    const first = balls[0];
    const second = balls.find((e) => e.detail !== first.detail);
    if (!second) return false;
    // 적구끼리 부딪히면 2적구가 그린 자리에 없다 — 계산한 줄이 아니게 된다.
    if (shot.events.some((e) => e.kind === "kiss" && e.at <= second.at)) return true;
    // 수구가 1적구를 한 번 더 건드리고 가는 것도 키스다. 실제로는 여기서
    // 진로가 막히거나 세기가 죽어 남은 길을 가지 못한다.
    return balls.some((e) => e !== first && e.detail === first.detail && e.at < second.at);
  }

  const strike = (ctx, deg, speed, side, vertical) => {
    const rad = deg * Math.PI / 180;
    return SIM.play(ctx.layout, ctx.cue, [Math.cos(rad) * speed, Math.sin(rad) * speed], side, vertical || 0);
  };

  // 일감 하나: [각도, 초속, 좌우, 상하, "sweep"이면 전방위 그물에서 나온 것].
  // → { promising, kiss, hit }  (hit은 득점하고 조건에 맞을 때만)
  function tryJob(ctx, job) {
    const [deg, speed, side, up, born] = job;
    const vertical = up || 0, fromSweep = born === "sweep";
    const { layout, cue } = ctx;
    const shot = strike(ctx, deg, speed, side, vertical);
    const judged = SIM.judge(shot);
    const balls = shot.events.filter((e) => e.kind === "ball");
    const cushions = shot.events.filter((e) => e.kind === "cushion");
    // 전방위 그물에서 나온 자리는 **쿠션을 먼저 맞은 것만**, 그리고 **쿠션 4개
    // 이상**일 때만 본다. 그게 이 그물을 친 이유(걸어치기·뱅크샷)이고 — 먼저
    // 하나 + 적구 사이 셋 = 넷이 필요하다 — 두께 사다리가 이미 나머지를 덮는다.
    // 이 조건이 없으면 될 만한 자리가 137개에서 381개로 불어나고, 그 각각이
    // 당점 13곳 × 각도 9칸으로 퍼져서 실행이 2.6초에서 18.1초가 된다.
    const promising = fromSweep
      ? (balls.length > 0 && cushions.length >= 4
         && cushions.some((e) => e.at < balls[0].at))
      : (balls.length > 0 && cushions.length >= 2);
    if (!judged.scored) {
      const gate = fromSweep ? ctx.gate : ctx.gateLadder;
      if (promising && gate !== null && balls.length
          && nearMiss(ctx, shot, balls[0].detail) > gate) return { promising: false, kiss: false, hit: null };
      return { promising, kiss: false, hit: null };
    }
    if (kissed(shot)) return { promising, kiss: true, hit: null };
    // 두께 = 1 − (수구 진행선과 적구 중심 사이 수직 거리) / 공 지름. 정면이 1,
    // 스치면 0. 진행선은 **1적구에 들어가는** 방향이다 — 쿠션을 먼저 맞는 샷은 처음
    // 겨냥과 다르다 (SIM.contact(), 2026-10-02).
    const { face, thickness } = SIM.contact(shot, layout, cue);
    if (ctx.wantFace !== "any" && face !== ctx.wantFace) return { promising, kiss: false, hit: null };
    if (ctx.wantFirst !== "any" && judged.first !== ctx.wantFirst) return { promising, kiss: false, hit: null };
    const { route, tags } = ROUTE.of(shot, judged, face, cue, ctx.L, ctx.W);
    if (!route) return { promising, kiss: false, hit: null };
    const way = shot.paths[judged.first] || [];
    let pushed = 0;
    for (let i = 1; i < way.length; i++) {
      pushed += Math.hypot(way[i][0] - way[i - 1][0], way[i][1] - way[i - 1][1]);
    }
    // 길의 이름표: 첫 3쿠션 레일 + 첫 쿠션 자리 칸. 화면의 finish()가 이것으로 묶는다.
    const bin = cushions.length
      ? `@${Math.floor(cushions[0].p[0] / BIN_MM)},${Math.floor(cushions[0].p[1] / BIN_MM)}` : "";
    const pathId = cushions.slice(0, 3).map((e) => e.detail).join("-") + bin;
    // 도는 쪽으로 회전을 주었는가 (+1 / −1 / 회전 없음 0). 동점을 가를 때 쓴다.
    const running = Math.abs(side || 0) < 0.01 ? 0
      : ((side > 0) === ROUTE.circuitIsRight(shot.paths[cue] || [], ctx.L, ctx.W) ? 1 : -1);
    return { promising: true, kiss: false,
             hit: { deg, speed, side, vertical, route, tags, face, pushed, thickness,
                    judged: { first: judged.first }, pathId, running } };
  }

  // 같은 공략이 되는 각도의 진짜 폭. 칸을 세지 않고 가장자리를 이분법으로 찾는다
  // (0.25도 눈금으로 세면 통의 95%가 같은 값에 묶였다).
  function sameLine(ctx, hit, deg) {
    const shot = strike(ctx, deg, hit.speed, hit.side, hit.vertical);
    const judged = SIM.judge(shot);
    if (!judged.scored || judged.first !== hit.judged.first || kissed(shot)) return false;
    const { face } = SIM.contact(shot, ctx.layout, ctx.cue);
    if (face !== hit.face) return false;
    const { route } = ROUTE.of(shot, judged, face, ctx.cue, ctx.L, ctx.W);
    return route === hit.route;
  }

  function trueRoom(ctx, hit) {
    let room = 0;
    for (const dir of [1, -1]) {
      let good = 0, bad = null;
      for (const span of [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]) {
        if (sameLine(ctx, hit, hit.deg + dir * span)) good = span; else { bad = span; break; }
      }
      if (bad === null) { room += good; continue; }
      for (let i = 0; i < 8; i++) {
        const middle = (good + bad) / 2;
        if (sameLine(ctx, hit, hit.deg + dir * middle)) good = middle; else bad = middle;
      }
      room += good;
    }
    return Math.round(room * 1000) / 1000;
  }

  // 일감 한 묶음. 순서를 지킨다 — 화면이 묶음을 차례로 이어 붙이면 한 줄로 돈 것과 같다.
  function runJobs(ctx, jobs) {
    const flags = new Array(jobs.length), hits = [];
    let kisses = 0;
    for (let i = 0; i < jobs.length; i++) {
      const got = tryJob(ctx, jobs[i]);
      flags[i] = got.promising ? 1 : 0;
      if (got.kiss) kisses++;
      if (got.hit) hits.push(got.hit);
    }
    return { flags, hits, kisses };
  }

  return { tryJob, runJobs, sameLine, trueRoom, kissed, nearMiss, strike };
})();

if (typeof importScripts === "function" && typeof document === "undefined") {
  self.onmessage = (event) => {
    const { id, kind, ctx, jobs, hits } = event.data;
    try {
      if (kind === "ping") self.postMessage({ id, ok: true });
      else if (kind === "jobs") self.postMessage({ id, ...SEARCH.runJobs(ctx, jobs) });
      else if (kind === "rooms") self.postMessage({ id, rooms: hits.map((h) => SEARCH.trueRoom(ctx, h)) });
    } catch (e) {
      self.postMessage({ id, error: String(e && e.message || e) });
    }
  };
}

if (typeof module !== "undefined") module.exports = SEARCH;
