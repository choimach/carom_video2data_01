// 워커로 돌린 계산이 화면에서 돌린 계산과 **똑같은지**, 그리고 얼마나 빨라졌는지.
//
//   node tools/check_workers.js <옛 빌드 폴더> build [씨앗,씨앗,…]
//
// 옛 빌드 폴더는 견줄 판 (carom_assistant.html + js들). 같은 폴더를 두 번 주면 화면 대 워커만 본다.
// 후보마다 각도·세기·당점·여유·점수·궤적 길이까지 견준다. 2026-10-03: 여섯 판 모두 같고 6~9초 → 1.3~2.2초.
const fs = require('fs'), path = require('path'), http = require('http');
const HOME = process.env.HOME;
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const { chromium } = require(core);
const [oldDir, newDir] = process.argv.slice(2, 4);
const seeds = (process.argv[4] || '3209700124,571926146,4171377588,3288843216,802329664,619503388').split(',').map(Number);
const serve = (dir) => new Promise((go) => {
  const s = http.createServer((q, r) => {
    const f = path.join(dir, q.url === '/' ? 'carom_assistant.html' : q.url.split('?')[0]);
    if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
    r.writeHead(200, { 'content-type': f.endsWith('.js') ? 'text/javascript' : 'text/html; charset=utf-8' });
    r.end(fs.readFileSync(f));
  }).listen(0, () => go(s));
});
(async () => {
  const b = await chromium.launch({ executablePath: path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
    env: { ...process.env, LD_LIBRARY_PATH: path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu') } });
  const sOld = await serve(oldDir), sNew = await serve(newDir);
  const runAll = async (port, workers) => {
    const p = await b.newPage();
    const errs = []; p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (m.type() === 'warning' || m.type() === 'error') errs.push(m.text()); });
    await p.goto(`http://localhost:${port}/`); await p.waitForTimeout(500); if (errs.length) console.log('페이지 오류', port, errs);
    const out = [];
    for (const seed of seeds) {
      const r = await p.evaluate(async ([seed, workers]) => {
        if (typeof USE_WORKERS !== 'undefined') USE_WORKERS = workers;
        stopPlaying(); result = null; picked = null; dealRandom(seed); render();
        const t0 = performance.now();
        // 화면이 얼마나 오래 막히는지: 16ms 박자가 얼마나 늦는지의 최댓값
        let worst = 0, last = performance.now();
        const tick = setInterval(() => { const n = performance.now(); worst = Math.max(worst, n - last); last = n; }, 16);
        const res = await advise(() => {});
        clearInterval(tick);
        return { ms: performance.now() - t0, worst, workers: typeof POOL !== 'undefined' ? POOL.count() : 0,
          rows: res.routes.map((r) => [r.key, r.hit.deg, r.hit.speed, r.hit.side, r.hit.vertical, r.hit.room, r.hit.rough, r.hit.lines,
                                       r.hit.thickness, r.hit.pushed, r.score, r.share, r.dupOf || null, r.hit.judged.rails.length, r.hit.shot.paths[cue].length]),
          kisses: res.kisses };
      }, [seed, workers]);
      out.push(r);
    }
    await p.close();
    return { out, errs };
  };
  const A = await runAll(sOld.address().port, false);
  const B = await runAll(sNew.address().port, false);
  const C = await runAll(sNew.address().port, true);
  const same = (x, y) => JSON.stringify([x.rows, x.kisses]) === JSON.stringify([y.rows, y.kisses]);
  seeds.forEach((seed, i) => {
    const a = A.out[i], n = B.out[i], w = C.out[i];
    console.log(`씨앗 ${seed}: 후보 ${a.rows.length} · 옛=새(화면) ${same(a, n)} · 옛=새(워커) ${same(a, w)} · 시간 옛 ${(a.ms / 1000).toFixed(1)} / 화면 ${(n.ms / 1000).toFixed(1)} / 워커${w.workers}개 ${(w.ms / 1000).toFixed(1)}초 · 화면 멈춤 최대 옛 ${Math.round(a.worst)} / 워커 ${Math.round(w.worst)} ms`);
  });
  console.log('오류', JSON.stringify([A.errs, B.errs, C.errs]).slice(0, 600));
  await b.close(); sOld.close(); sNew.close();
})();
