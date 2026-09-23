// 여유를 0.05도로 재면 바닥값에 묶여 있던 통들이 갈리는가.
//
// 지금 눈금은 0.25도인데, 통 35,742개 중 33%가 그 바닥값에 있다. 순위에서
// "얼마나 쉬운가"를 맡은 값이 세 번에 한 번은 아무 말도 못 한다는 뜻이다.
// 첫 무작위 판(씨앗 1581786015)에서는 후보 12개 중 11개가 바닥값이었고,
// 0.05도로 다시 재니 0.00~0.15도로 갈렸다.
//
//     node tools/measure_room_step.js 20
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const { search, SIM, FINE } = require(path.join(ROOT, 'tools', 'enumerate_alternatives.js'));
const ROUTE = require(path.join(ROOT, 'src', 'visualization', 'assistant', 'route.js'));

function kissed(shot) {
  const balls = shot.events.filter((e) => e.kind === 'ball');
  if (!balls.length) return false;
  const first = balls[0];
  const second = balls.find((e) => e.detail !== first.detail);
  if (!second) return false;
  if (shot.events.some((e) => e.kind === 'kiss' && e.at <= second.at)) return true;
  return balls.some((e) => e !== first && e.detail === first.detail && e.at < second.at);
}

// 대표 줄 하나의 진짜 창.
//
// 칸을 세지 않고 **가장자리를 이분법으로 찾는다.** 0.05도 칸으로 세어 보면
// 창이 대개 0.05~0.15도라 값이 세 가지밖에 안 나와 또 묶인다. 배로 늘려
// 가장자리를 가둔 다음 여덟 번 반으로 접으면 같은 30번 계산으로 0.006도까지
// 나오고, 값이 연속이라 묶이지 않는다.
const STEP = 0.05;
function edgeRoom(layout, cue, h, ok) {
  let room = 0;
  for (const dir of [1, -1]) {
    let good = 0, bad = null;
    for (const span of [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]) {
      if (ok(h.deg + dir * span)) { good = span; } else { bad = span; break; }
    }
    if (bad === null) { room += good; continue; }   // 3.2도까지 열려 있다
    for (let i = 0; i < 8; i++) {
      const mid = (good + bad) / 2;
      if (ok(h.deg + dir * mid)) good = mid; else bad = mid;
    }
    room += good;
  }
  return Math.round(room * 1000) / 1000;
}

function fineRoom(layout, cue, h) {
  const from = layout[cue];
  const v = SIM.speedFor(h.strength);
  const key = `${h.route}|${h.first}|${h.face}`;
  const ok = (deg) => {
    const rad = deg * Math.PI / 180, aim = [Math.cos(rad), Math.sin(rad)];
    const shot = SIM.play(layout, cue, [aim[0] * v, aim[1] * v], h.side, h.up);
    shot.cue = cue;
    const judged = SIM.judge(shot);
    if (!judged.scored || kissed(shot)) return false;
    const target = layout[judged.first];
    const across = aim[0] * (target[1] - from[1]) - aim[1] * (target[0] - from[0]);
    const face = across > 0 ? 'left' : 'right';
    const { route } = ROUTE.of(shot, judged, face, cue, SIM.L, SIM.W);
    return `${route}|${judged.first}|${face}` === key;
  };
  let room = 0;
  for (const dir of [1, -1]) {
    for (let k = (dir > 0 ? 0 : 1); k < 60; k++) {
      if (!ok(h.deg + dir * k * STEP)) break;
      room += STEP;
    }
  }
  return { step: Math.round(room * 100) / 100, edge: edgeRoom(layout, cue, h, ok) };
}

// 조언판 finish()와 같은 대표 뽑기: 여유가 가장 넓은 줄, 동점이면 같은 당점으로
// 득점하는 줄이 많은 쪽, 그래도 같으면 도는 쪽으로 회전을 준 줄.
function representatives(hits) {
  const key = (h, deg) => `${h.speed}:${h.side}:${Math.round(deg / FINE) * FINE}`;
  const at = new Map();
  for (const h of hits) at.set(key(h, h.deg), h);
  for (const h of hits) {
    let room = FINE;
    for (const dir of [FINE, -FINE]) {
      let d = h.deg + dir;
      while (at.has(key(h, (d + 360) % 360))) { room += FINE; d += dir; }
    }
    h.room = Math.round(room * 100) / 100;
  }
  const idOf = (h) => `${h.route}|${h.first}|${h.face}`;
  const tipOf = (h) => `${Math.round((h.side||0)*100)}:${Math.round((h.up||0)*100)}`;
  const perTip = new Map();
  for (const h of hits) {
    const t = `${idOf(h)}@${tipOf(h)}`;
    perTip.set(t, (perTip.get(t) || 0) + 1);
  }
  const best = new Map();
  for (const h of hits) {
    const id = idOf(h);
    const kept = best.get(id);
    if (!kept || h.room > kept.room
        || (h.room === kept.room
            && perTip.get(`${id}@${tipOf(h)}`) > perTip.get(`${id}@${tipOf(kept)}`))) {
      best.set(id, h);
    }
  }
  return best;
}

const rounds = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l));
const N = Number(process.argv[2] || 20);
const step = Math.max(1, Math.floor(rounds.length / N));
const pick = [];
for (let i = 0; i < rounds.length && pick.length < N; i += step) pick.push(rounds[i]);

let tiedCoarse = 0, tiedFine = 0, buckets = 0, layouts = 0;
const fineAll = [], edgeAll = [], tiedEdge = [];
for (const one of pick) {
  const hits = search(one.layout, one.cue);
  if (!hits.length) continue;
  layouts++;
  const best = representatives(hits);
  const coarse = [], fine = [];
  for (const [, h] of best) {
    coarse.push(h.room);
    const f = fineRoom(one.layout, one.cue, h);
    fine.push(f.step); fineAll.push(f.edge); edgeAll.push(f.edge);
    buckets++;
  }
  // 같은 값에 묶인 통이 몇 개인가 (1등을 가리지 못하는 통)
  const count = (a) => { const c = new Map(); for (const x of a) c.set(x, (c.get(x) || 0) + 1);
    return [...c.values()].filter((n) => n > 1).reduce((s, n) => s + n, 0); };
  tiedCoarse += count(coarse); tiedFine += count(fine);
  tiedEdge.push(count(edgeAll.slice(-best.size)));
}
const mid = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
console.log(`프로 배치 ${layouts}개 · 통 ${buckets}개\n`);
console.log(`  0.25도 눈금에서 같은 값에 묶인 통  ${tiedCoarse} (${(tiedCoarse/buckets*100).toFixed(0)}%)`);
console.log(`  0.05도 눈금에서 같은 값에 묶인 통  ${tiedFine} (${(tiedFine/buckets*100).toFixed(0)}%)`);
const tiedE = tiedEdge.reduce((a, b) => a + b, 0);
console.log(`  이분법(0.006도)에서 묶인 통       ${tiedE} (${(tiedE/buckets*100).toFixed(0)}%)`);
console.log(`\n  실제 창 중앙값 ${mid(fineAll).toFixed(2)}도`
  + ` · 가장 넓은 것 ${Math.max(...fineAll).toFixed(2)}도`
  + ` · 0.05도 이하 ${(fineAll.filter((x) => x <= 0.05).length / fineAll.length * 100).toFixed(0)}%`);
