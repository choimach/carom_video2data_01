// 첫 쿠션을 지난 뒤 궤적이 얼마나 어긋나는가 — 선택을 빼고 물리만 잰다.
//
// 왜 이 모양인가. 조언판의 오차에는 두 가지가 섞여 있다 — **무엇을 칠지
// 고른 것**과 **고른 것을 얼마나 맞게 그리는지**. 여기서는 프로가 실제로 친
// 겨냥과 강도를 그대로 쓰므로 고르기가 빠진다.
//
// 당점은 영상에서 재지 못한다. 그래서 **첫 쿠션 자리로 당점을 맞춘다** —
// 팁 격자에서 첫 쿠션이 가장 잘 맞는 것을 고르고, 그 다음 **둘째 쿠션이
// 얼마나 빗나가는지** 본다. 첫 쿠션까지는 이미 맞으므로(113 mm), 여기서
// 남는 것은 전부 "쿠션을 지난 뒤"의 문제다.
//
//   node tools/check_after_cushion.js [판 수]

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

const atClock = (h, t) => [t * Math.sin(h / 12 * 2 * Math.PI), t * Math.cos(h / 12 * 2 * Math.PI)];
const TIPS = [[0, 0]];
for (const h of [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) {
  for (const t of [1, 2, 3]) TIPS.push(atClock(h, t));
}

const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const plays = data.plays.filter((p) => p.aim !== null && p.speed
  && p.rails && p.rails.length >= 2);

const N = Number(process.argv[2] || 150);
const step = Math.max(1, Math.floor(plays.length / N));
const pick = [];
for (let i = 0; i < plays.length && pick.length < N; i += step) pick.push(plays[i]);

const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const mid = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const first = [], second = [], third = [], bestSecond = [];
for (const play of pick) {
  const layout = { white: play.cue, yellow: play.balls[0], red: play.balls[1] };
  const rad = play.aim * Math.PI / 180;
  const aim = [Math.cos(rad) * play.speed, Math.sin(rad) * play.speed];
  let pinned = null, loose = null;
  for (const [side, up] of TIPS) {
    const shot = SIM.play(layout, 'white', aim, side, up);
    const cs = shot.events.filter((e) => e.kind === 'cushion').map((e) => e.p);
    if (cs.length < 1) continue;
    const one = gap(cs[0], play.rails[0]);
    if (!pinned || one < pinned.one) pinned = { one, cs };
    if (cs.length >= 2) {
      const two = gap(cs[1], play.rails[1]);
      if (!loose || two < loose.two) loose = { two };
    }
  }
  if (!pinned) continue;
  first.push(pinned.one);
  if (pinned.cs.length >= 2) second.push(gap(pinned.cs[1], play.rails[1]));
  if (pinned.cs.length >= 3 && play.rails.length >= 3) third.push(gap(pinned.cs[2], play.rails[2]));
  if (loose) bestSecond.push(loose.two);
}

console.log(`프로 플레이 ${pick.length}개 (겨냥·강도·쿠션 자리가 다 있는 것)\n`);
console.log('첫 쿠션으로 당점을 맞춘 뒤, 쿠션 자리가 영상과 얼마나 어긋나나 (중앙값)');
console.log(`  1쿠션 (맞춘 것이므로 이만큼이 한계)   ${Math.round(mid(first))} mm   n=${first.length}`);
console.log(`  2쿠션                              ${Math.round(mid(second))} mm   n=${second.length}`);
if (third.length) console.log(`  3쿠션                              ${Math.round(mid(third))} mm   n=${third.length}`);
console.log(`\n  참고: 당점을 2쿠션에 맞췄을 때의 2쿠션 오차  ${Math.round(mid(bestSecond))} mm`);
console.log('  (이것보다 위의 2쿠션 값이 훨씬 크면, 첫 쿠션에 맞는 당점과');
console.log('   둘째 쿠션에 맞는 당점이 서로 다르다는 뜻 — 쿠션 뒤 처리가 틀린 것이다.)');
