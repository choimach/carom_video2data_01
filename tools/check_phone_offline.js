// 폰 앱의 "결과 보내기"가 오프라인에서도 살아남는지: 온라인 보내기 → 오프라인 보내기 → 닫기 → 다시 열기.
// 실제 주소(carom001-board)에 [자동 시험] 기록을 쓴다 — 끝나면 Firestore에서 지울 것 (pull_feedback.py 아래 참고).
// 2026-10-03: 다시 연 뒤 25초 안에 올라감 (mutations 1 → 0).
const fs = require('fs'), path = require('path');
const HOME = process.env.HOME;
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const { chromium } = require(core);
(async () => {
  const b = await chromium.launch({ executablePath: path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
    env: { ...process.env, LD_LIBRARY_PATH: path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu') } });
  const ctx = await b.newContext({ viewport: { width: 412, height: 915 }, isMobile: true, hasTouch: true });
  const errs = [];
  const open = async () => { const p = await ctx.newPage(); p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (/firestore|firebase|Firestore/i.test(m.text())) errs.push(m.type() + ': ' + m.text().slice(0, 200)); }); await p.goto('https://carom001-board.web.app/'); await p.waitForFunction(() => window.CAROM_CLOUD, null, { timeout: 20000 }); return p; };
  const send = async (p, note) => {
    await p.fill('#seed', '619503388'); await p.press('#seed', 'Enter');
    await p.waitForFunction(() => result && document.getElementById('run').textContent === '실행', null, { timeout: 90000 });
    await p.fill('#comment', note); await p.tap('#copy');
    await p.waitForFunction(() => /보냈|저장|실패/.test(document.getElementById('copied').textContent), null, { timeout: 15000 });
    return p.textContent('#copied');
  };
  const p1 = await open();
  console.log('온라인:', await send(p1, '[자동 시험] 온라인'));
  await ctx.setOffline(true);
  console.log('오프라인:', await send(p1, '[자동 시험] 오프라인에서 보냄'));
  console.log('IDB', await p1.evaluate(async () => (await indexedDB.databases()).map((d) => d.name))); await p1.waitForTimeout(3000); await p1.close();
  await ctx.setOffline(false);
  const p2 = await open(); await p2.addScriptTag({ path: require('path').join(__dirname, 'idb_probe.js') }); console.log('다시 연 직후', JSON.stringify(await p2.evaluate(() => __probe())));
  await p2.waitForTimeout(25000); console.log('25초 뒤', JSON.stringify(await p2.evaluate(() => __probe())));
  console.log('다시 엶 · 오류', JSON.stringify(errs));
  await b.close();
})();
