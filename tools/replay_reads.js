// check_opening_read.py가 부른다: 두 읽기(옛·새)로 프로 샷을 다시 쳐서 재현율을 낸다.
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');
const plays = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const gap = (a, b) => (Math.hypot(a[0][0]-b[0][0], a[0][1]-b[0][1]) + Math.hypot(a[1][0]-b[1][0], a[1][1]-b[1][1])) / 2;
const TIPS = []; for (const s of [-2, -1, 0, 1, 2]) for (const u of [-1, 0, 1, 2]) TIPS.push([s, u]);
function replay(p, read) {
  const [aim, speed] = read;
  const rad = aim * Math.PI / 180, v = [Math.cos(rad) * speed, Math.sin(rad) * speed];
  let top = null;
  for (const [s, u] of TIPS) {
    const shot = SIM.play(p.layout, p.cue, v, s, u);
    const c = shot.events.filter((e) => e.kind === 'cushion').slice(0, 2).map((e) => e.p);
    if (c.length < 2) continue;
    const g = gap(c, p.rails);
    if (!top || g < top.g) top = { g, first: SIM.judge(shot).first };
  }
  return top;
}
const both = plays.filter((p) => p.new);
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s[s.length >> 1]; };
for (const which of ['old', 'new']) {
  const g = [], first = [];
  for (const p of both) {
    const r = replay(p, p[which]);
    first.push(r && r.first === p.first ? 1 : 0);
    g.push(r ? r.g : 99999);
  }
  const pct = (v) => Math.round(100 * v.reduce((a, b) => a + b, 0) / v.length);
  console.log(`${which === 'old' ? '옛 읽기 (1~5프레임)' : '새 읽기 (첫 사건 전 + 되풀기)'}: ${both.length}판 · 1적구 같음 ${pct(first)}%` +
    ` · 첫 두 쿠션 300 안 ${pct(g.map((x) => (x <= 300 ? 1 : 0)))}% · 중앙값 ${Math.round(median(g))} mm`);
}
