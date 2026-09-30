// 조언판 한 배치가 어디에 시간을 쓰나 — 단계별 시간과 시뮬 호출 수.
//
// 빌드된 페이지를 헤드리스 크롬으로 열어 프로 배치 몇 개에 `advise()`를 부른다.
// 단계는 solve()의 진행률로 가른다: 0~0.45 훑기(두께 사다리 + 전방위), 0.45~0.70 다듬기,
// 0.70~0.80 당점 양, 0.80~1 씨앗. finish()(묶기 + 여유 정밀), fromPros()(이웃)는 감싸서 잰다.
// onProgress 한 번 = await pause() 한 번 (setTimeout 0 — 브라우저가 쉬는 시간).
//
//   node tools/profile_app.js [배치 수=8]
const fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname), HOME = process.env.HOME;
const N = Number(process.argv[2] || 8);
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const SHELL = path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell');
const LIBS = path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu');
const { chromium } = require(core);

const lines = fs.readFileSync(path.join(ROOT, 'data', 'alternatives.jsonl'), 'utf8').split('\n').filter(Boolean);
const step = Math.floor(lines.length / N);
const plays = lines.filter((_, i) => i % step === 0).slice(0, N).map((l) => JSON.parse(l));

(async () => {
  const browser = await chromium.launch({ executablePath: SHELL, env: { ...process.env, LD_LIBRARY_PATH: LIBS } });
  const page = await browser.newPage();
  await page.goto('file://' + path.join(ROOT, 'build', 'carom_assistant.html'));
  // PRE: 페이지를 연 뒤 돌릴 JS — 변형을 시험할 때 (예: PRE='REFINE_TIPS = TIP_POINTS.slice(0, 5)').
  if (process.env.PRE) await page.evaluate(process.env.PRE);
  await page.evaluate(() => {
    window.PROF = { plays: 0, playMs: 0 };
    const play = SIM.play;
    SIM.play = (...a) => { const t = performance.now(); const r = play(...a); PROF.plays++; PROF.playMs += performance.now() - t; return r; };
    const wrap = (name) => { const f = window[name]; window[name] = (...a) => { const t = performance.now(); const r = f(...a); PROF[name] = (PROF[name] || 0) + performance.now() - t; return r; }; };
    wrap('finish'); wrap('fromPros'); wrap('trueRoom');
  });
  const total = {};
  for (const p of plays) {
    const got = await page.evaluate(async (p) => {
      layout = JSON.parse(JSON.stringify(p.layout)); cue = p.cue; wantFirst = 'any'; wantFace = 'any'; excludeMatch = p.match.replace('soop_', '');
      for (const k of Object.keys(PROF)) PROF[k] = 0;
      const marks = []; const t0 = performance.now();
      const r = await advise((done) => marks.push([done, performance.now()]));
      const end = performance.now();
      const stage = (lo, hi) => { const m = marks.filter(([d]) => d >= lo && d < hi); return m.length ? m[m.length - 1][1] - (marks.filter(([d]) => d < lo).pop() || [0, t0])[1] : 0; };
      const st = (window.solve && window.solve.stats) || {};
      return { total: end - t0, pauses: marks.length, routes: r.routes.length, ...PROF, st_contact: st.contact, st_promising: st.promising, st_sweep: st.fromSweep, st_around: st.around,
               s1: stage(0, 0.45), s2: stage(0.45, 0.70), s3: stage(0.70, 0.80), s4: stage(0.80, 1.01) };
    }, p);
    for (const [k, v] of Object.entries(got)) total[k] = (total[k] || 0) + v;
  }
  const avg = (k) => (total[k] || 0) / plays.length;
  const ms = (k) => `${(avg(k) / 1000).toFixed(2)}초`;
  console.log(`배치 ${plays.length}개 평균 — 전체 ${ms('total')} · 후보 ${avg('routes').toFixed(0)}개`);
  console.log(`  시뮬 호출 ${avg('plays').toFixed(0)}번 · 그 시간 ${ms('playMs')} (${(avg('playMs') / avg('plays')).toFixed(2)} ms/번)`);
  console.log(`  ① 훑기 ${ms('s1')} · ② 다듬기 ${ms('s2')} · ③ 당점 양 ${ms('s3')} · ④ 씨앗 ${ms('s4')}`);
  console.log(`  finish (묶기+여유) ${ms('finish')} — 그중 trueRoom ${ms('trueRoom')} · fromPros ${ms('fromPros')}`);
  console.log(`  훑기 시도 ${avg('st_contact').toFixed(0)} · 가망 ${avg('st_promising').toFixed(0)} (전방위 출신 ${avg('st_sweep').toFixed(0)}) · 다듬기 시도 ${avg('st_around').toFixed(0)}`);
  console.log(`  쉬기(pause) ${avg('pauses').toFixed(0)}번`);
  await browser.close();
})();
