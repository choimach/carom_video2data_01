// 놓친 판은 왜 놓쳤나 — 격자가 성긴 탓인가, 물리·이름 탓인가.
//
// 프로가 친 (공, 면)만 붙들고 각도 0.05도 · 당점 37곳 · 강도 9가지로 다
// 뒤진다. 그러고도 못 찾으면 격자 문제가 아니다.
//
// 2026-09-24 실측 (옛 물리, 놓친 판 30개):
//     촘촘히 뒤져서 찾음   50%   ← 격자가 성긴 탓
//     그래도 못 찾음       50%   ← 물리나 이름
//
// ★못 찾은 쪽에서도 같은 (공, 면)으로 득점하는 줄은 나온다. **이름만 다르게
//   붙는다** (대회전 12 · 옆돌리기 12 · 횡단 10 · 뒤돌리기 10 · 빗겨치기 5).
//   "못 찾았다"가 아니라 "찾았는데 다른 이름을 붙였다"이다.
//
//   node tools/why_missed.js [유형] [판 수]
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');
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
const atClock = (h, t) => [t * Math.sin(h / 12 * 2 * Math.PI), t * Math.cos(h / 12 * 2 * Math.PI)];
const TIPS = [[0, 0]];
for (const h of [12,1,2,3,4,5,6,7,8,9,10,11]) for (const t of [1,2,3]) TIPS.push(atClock(h, t));
const SPEEDS = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7].map((s) => SIM.speedFor(s));

const rounds = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l));
const missed = rounds.filter((r) => !r.reached);
const want = process.argv[2] || null;
const pick = (want ? missed.filter((r) => r.chose.startsWith(want + '|')) : missed)
  .filter((_, i) => i % 3 === 0).slice(0, Number(process.argv[3] || 30));

let found = 0, names = new Map();
for (const one of pick) {
  const [route, first, face] = one.chose.split('|');
  const layout = one.layout, cue = one.cue, from = layout[cue], to = layout[first];
  const straight = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI;
  const reach = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const swing = Math.asin(Math.min(1, SIM.DIAMETER / Math.max(reach, SIM.DIAMETER))) * 180 / Math.PI;
  const got = new Set();
  let hit = false;
  outer:
  for (const [side, up] of TIPS) {
    for (const v of SPEEDS) {
      for (let d = -swing - 0.3; d <= swing + 0.3; d += 0.05) {
        const deg = straight + d, rad = deg * Math.PI / 180;
        const aim = [Math.cos(rad), Math.sin(rad)];
        const shot = SIM.play(layout, cue, [aim[0] * v, aim[1] * v], side, up);
        shot.cue = cue;
        const judged = SIM.judge(shot);
        if (!judged.scored || judged.first !== first || kissed(shot)) continue;
        const across = aim[0] * (to[1] - from[1]) - aim[1] * (to[0] - from[0]);
        const f = across > 0 ? 'left' : 'right';
        if (f !== face) continue;
        const { route: r } = ROUTE.of(shot, judged, f, cue, SIM.L, SIM.W);
        if (!r) continue;
        got.add(r);
        if (r === route) { hit = true; break outer; }
      }
    }
  }
  if (hit) found++;
  else for (const r of got) names.set(r, (names.get(r) || 0) + 1);
}
console.log(`놓친 판 ${pick.length}개를 촘촘히 다시 뒤짐${want ? ' (' + want + ')' : ''}`);
console.log(`  더 촘촘히 뒤져서 찾은 것        ${found} (${(found/pick.length*100).toFixed(0)}%)  ← 해상도 문제`);
console.log(`  그래도 못 찾은 것              ${pick.length - found}  ← 물리나 이름의 문제`);
if (names.size) console.log('  못 찾은 판에서 같은 (공, 면)으로 나온 이름:',
  [...names].sort((a,b)=>b[1]-a[1]).map(([r,n])=>`${r} ${n}`).join(' · '));
