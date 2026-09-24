// 당점 오차 중 무엇이 공략을 무너뜨리나 — 좌우인가 상하인가.
//
// 2026-09-24 실측 (프로가 친 줄 200개, 실제 득점률 61%):
//
//     당점 오차    좌우만   상하만   둘 다
//       0.1팁      71.5%   88.4%   66.8%
//       0.2팁      56.8%   81.6%   50.6%
//       0.3팁      47.2%   76.8%   40.5%
//       0.5팁      36.6%   70.9%   30.5%
//
// ★좌우가 범인이다. 상하(밀어·끌어)는 훨씬 너그럽다.
//
// ⚠️ 쿠션의 회전 모델 자체는 멀쩡하다. 한때 "1팁에서 포화한다"고 적었다가
// 물렀다 — 시험에서 적구를 쏘는 쿠션 바로 옆에 놓는 바람에 수구가 쿠션이
// 아니라 공을 맞고 있었다. 깨끗이 다시 재면 팁당 5~7도로 고르게 반응하고
// (20도 입사에서 3팁이 −27도), 영상의 거울 대비 틀어짐(중앙값 10.8도,
// 90% 27.1도)과 잘 맞는다.
//
// 그러므로 남은 용의자는 **첫 쿠션을 지난 뒤의 궤적**이다. 첫 쿠션 자리는
// 영상과 113 mm로 맞는데 둘째 쿠션에서 370 mm로 벌어진다. 우리가 재는
// "조준 창"은 진짜 창이 아니라 우리 오차의 폭일 수 있다.
//
//   node tools/tip_sensitivity.js
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const { SIM } = require(path.join(ROOT, 'tools', 'make_probability.js'));

function kissed(shot) {
  const balls = shot.events.filter((e) => e.kind === 'ball');
  if (!balls.length) return false;
  const first = balls[0];
  const second = balls.find((e) => e.detail !== first.detail);
  if (!second) return false;
  if (shot.events.some((e) => e.kind === 'kiss' && e.at <= second.at)) return true;
  return balls.some((e) => e !== first && e.detail === first.detail && e.at < second.at);
}
function rng(seed) { let s = (seed >>> 0) || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function gauss(n) { const u = Math.max(n(), 1e-12), v = n(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

const rounds = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l)).filter((r) => r.reached);
const N = 200, step = Math.max(1, Math.floor(rounds.length / N));
const lines = [];
for (let i = 0; i < rounds.length && lines.length < N; i += step) {
  const one = rounds[i];
  const b = one.found.find((x) => x.key === one.chose);
  if (!b || b.deg === undefined) continue;
  lines.push({ layout: one.layout, cue: one.cue,
    line: { deg: b.deg, speed: SIM.speedFor(b.strength), side: b.side, up: b.up } });
}
function share(dSide, dUp, tries = 40) {
  let made = 0, all = 0;
  lines.forEach((x, k) => {
    const n = rng(k + 1);
    for (let i = 0; i < tries; i++) {
      const side = x.line.side + gauss(n) * dSide;
      const up = (x.line.up || 0) + gauss(n) * dUp;
      const out = Math.hypot(side, up);
      const keep = out > SIM.MAX_TIPS ? SIM.MAX_TIPS / out : 1;
      const rad = x.line.deg * Math.PI / 180;
      const shot = SIM.play(x.layout, x.cue,
        [Math.cos(rad) * x.line.speed, Math.sin(rad) * x.line.speed], side * keep, up * keep);
      shot.cue = x.cue;
      const j = SIM.judge(shot);
      if (j.scored && !kissed(shot)) made++;
      all++;
    }
  });
  return made / all;
}
console.log(`프로가 친 줄 ${lines.length}개 · 실제 득점률 61%\n`);
console.log('당점 오차       좌우만    상하만    둘 다');
for (const d of [0.1, 0.2, 0.3, 0.5]) {
  console.log(`  ${d}팁        ${(share(d,0)*100).toFixed(1)}%    ${(share(0,d)*100).toFixed(1)}%    ${(share(d,d)*100).toFixed(1)}%`);
}
