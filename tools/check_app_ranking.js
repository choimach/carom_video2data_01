// 조언판의 순위를 **조언판 코드로** 잰다 — 파이썬(learn_choices.py)과 같은 표.
//
// 왜 (2026-10): 순위는 파이썬으로만 재 왔다. 그 사이 조언판에서만 난 버그가 둘 나왔고
// (app-data의 면이 절반 뒤집힘, 이웃 투표에 실제 틀의 면을 씀) 둘 다 파이썬 점수에는
// 안 보였다. 여기서는 빌드된 페이지를 헤드리스 크롬으로 열어 가려 둔 프로 판마다
// `advise()`를 부르고, 상위 3개 후보의 첫 두 쿠션 자리를 프로가 실제로 간 자리
// (app-data `rails`)와 견준다. 같은 경기의 판은 이웃에서 뺀다 (`excludeMatch`).
//
// 배치는 model.json의 정규화된 틀 그대로 넣는다 — 그 틀에서는 수구가 왼쪽 아래
// 사분면이라 조언판의 frameOf()가 아무것도 안 뒤집고, 프로의 rails와 바로 견줄 수 있다.
//
//   node tools/check_app_ranking.js [판 수=120]
//
// 필요한 것: build/carom_assistant.html (tools/build_assistant.py), 헤드리스 크롬
// (tools/screenshot_assistant.sh가 라이브러리를 받아 둔다 — 먼저 한 번 돌릴 것).
const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const HOME = process.env.HOME;
const N = Number(process.argv[2] || 120);
// 페이지를 몇 개 동시에 — 판당 16초라 하나씩이면 120판에 32분이다. 코어 절반 규칙
// (CLAUDE.md §3.5) 안에서 넷.
const PAGES = Number(process.env.PAGES || 4);

const find = (dir, test) => {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (test(full)) return full;
  }
  return null;
};
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core'))
  .find((p) => fs.existsSync(p));
const shellRoot = find(path.join(HOME, '.cache', 'ms-playwright'), (p) => path.basename(p).startsWith('chromium_headless_shell-'));
const SHELL = path.join(shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell');
const LIBS = path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu');
const { chromium } = require(core);

// 가려 둘 판: 탐색이 프로의 길을 찾았고 프로 쿠션이 둘 넘게 찍힌 판을 경기에 고르게.
const app = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'app-data.json'), 'utf8'));
const railsOf = new Map(app.plays.map((p) => [`soop_${p.match}:${p.inning}:${p.shot}`, p.rails || []]));
const all = [];
for (const line of fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8').split('\n')) {
  if (!line) continue;
  const one = JSON.parse(line);
  const rails = railsOf.get(one.id) || [];
  if (one.reached && rails.length >= 2) all.push({ id: one.id, match: one.match.replace('soop_', ''), cue: one.cue, layout: one.layout, rails });
}
const step = Math.max(1, Math.floor(all.length / N));
const plays = all.filter((_, i) => i % step === 0).slice(0, N);

const gap = (a, b) => (Math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]) + Math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1])) / 2;
const median = (v) => { const s = [...v].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

(async () => {
  const browser = await chromium.launch({ executablePath: SHELL, env: { ...process.env, LD_LIBRARY_PATH: LIBS } });
  const errors = [];
  const top1 = [], top3 = [], started = Date.now(), perPlay = {}, english = {};
  let next = 0, done = 0;
  const worker = async () => {
  const page = await browser.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('file://' + path.join(ROOT, 'build', 'carom_assistant.html'));
  // PRE: 페이지를 연 뒤 돌릴 JS — 변형을 시험할 때 (예: PRE='REFINE_TIPS = TIP_POINTS.slice(0, 5)').
  if (process.env.PRE) await page.evaluate(process.env.PRE);
  while (next < plays.length) {
    const play = plays[next++];
    const got = await page.evaluate(async ({ play }) => {
      layout = JSON.parse(JSON.stringify(play.layout));
      cue = play.cue; wantFirst = 'any'; wantFace = 'any'; excludeMatch = play.match;
      const flips = frameOf();
      const r = await advise(() => {});
      excludeMatch = null;
      return {
        flipped: flips.flipX || flips.flipY,
        top: r.routes.slice(0, 3).map((row) => row.hit.shot.events
          .filter((e) => e.kind === 'cushion').slice(0, 2).map((e) => e.p)),
        // 1등 줄의 회전이 도는 방향과 같은가 — 선수: "좌우가 바뀌는 것이 가장 흔한 문제".
        // 프로는 73%가 도는 쪽으로 준다 (tools/check_side_vs_circuit.py).
        english: r.routes.length ? (() => {
          const h = r.routes[0].hit;
          if (Math.abs(h.side || 0) < 0.3) return 'none';
          return ((h.side > 0) === ROUTE.circuitIsRight(h.shot.paths[cue] || [], L, W)) ? 'running' : 'reverse';
        })() : null,
      };
    }, { play });
    if (got.flipped) throw new Error(`${play.id}: 정규화된 배치인데 frameOf()가 뒤집는다 — 틀이 다르다`);
    const gaps = got.top.filter((c) => c.length >= 2).map((c) => gap(c, play.rails));
    if (got.english) english[got.english] = (english[got.english] || 0) + 1;
    if (gaps.length) { top1.push(gaps[0]); top3.push(Math.min(...gaps)); perPlay[play.id] = [gaps[0], Math.min(...gaps)]; }
    done++;
    if (done % 20 === 0) console.log(`  ${done}/${plays.length} · 판당 ${((Date.now() - started) / 1000 / done).toFixed(1)}초`);
  }
  };
  await Promise.all(Array.from({ length: PAGES }, worker));
  await browser.close();
  // DUMP: 판별 (1등, 상위 3개 최선) 거리를 JSON으로 — 두 변형을 짝지어 견줄 때.
  if (process.env.DUMP) fs.writeFileSync(process.env.DUMP, JSON.stringify(perPlay));
  const within = (v, mm) => `${Math.round(100 * v.filter((x) => x <= mm).length / v.length)}%`;
  console.log(`조언판으로 잰 순위 — ${top1.length}판 (같은 경기는 이웃에서 뺐다), 오류 ${errors.length}`);
  console.log(`  1등 후보          중앙값 ${Math.round(median(top1))} mm · 300 안 ${within(top1, 300)} · 500 안 ${within(top1, 500)}`);
  console.log(`  상위 3개 중 최선   중앙값 ${Math.round(median(top3))} mm · 300 안 ${within(top3, 300)} · 500 안 ${within(top3, 500)}`);
  const turned = (english.running || 0) + (english.reverse || 0);
  if (turned) console.log(`  1등 줄의 회전: 도는 쪽 ${english.running || 0} · 반대쪽 ${english.reverse || 0}`
    + ` (${Math.round(100 * (english.running || 0) / turned)}% 도는 쪽, 프로 73%) · 거의 없음 ${english.none || 0}`);
  if (errors.length) console.log('  ' + errors.slice(0, 3).join('\n  '));
})();
