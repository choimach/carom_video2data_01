// 선수가 판단한 배치를 **지금의 조언판**으로 다시 풀어, 판단이 아직 지켜지는지 센다 (2026-10-07).
//
// 선수: "맞다로 결과 나온 것도 중요한 것 아니야?" — 그렇다. `tools/table_check.py`는 판단을 받은 **그때의**
// 순위만 채점한다. 그래서 순위를 고칠 때마다 (뱅크샷 문, 역회전) 불평은 고쳐졌는지 봤지만, 예전에 "맞다"를
// 받은 줄이 상위에서 밀려났는지는 아무도 안 봤다. 이 도구가 그것을 잰다:
//
//   맞다 받은 줄 — 지금도 3등 안인가 (그때는 몇 등이었나)
//   아니다·낮음 받은 줄 — 지금 3등 안에 들어와 있지는 않은가
//   지금의 1등 — 선수가 판단한 줄인가, 무엇이었나
//
// 순위를 바꾸는 변경은 이 숫자가 **나빠지지 않을 때만** 올린다.
//
//   node tools/check_verdicts.js                 # 지금 build/의 조언판
//   PAGE=다른/carom_assistant.html node tools/check_verdicts.js     # 다른 판과 견줄 때
//   node tools/check_verdicts.js --save 파일.json  # 판별 결과를 남겨 두 판을 짝지어 본다
//
// 배치는 기록의 숫자 여섯 개로 놓고 수구를 누른다 (씨앗보다 확실하다 — 씨앗을 까는 코드가 바뀌어도 같은 배치).
// 줄 맞추기: 열쇠(유형|공|면|길@칸)가 같으면 그 줄, 아니면 칸만 다른 줄, 아니면 유형|공|면이 같은 가장 높은 줄.

const fs = require('fs');
const path = require('path');
const ROOT = path.dirname(__dirname);
const HOME = process.env.HOME;

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

const saveAt = process.argv.includes('--save') ? process.argv[process.argv.indexOf('--save') + 1] : null;
const ledger = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'table_feedback.json'), 'utf8')).rounds
  .filter((r) => r.layout && r.cue && r.verdicts && Object.keys(r.verdicts).length);

const loose = (key) => key.split('@')[0];
const family = (key) => key.split('|').slice(0, 3).join('|');
// 같은 유형·공·면·쿠션 차례면 같은 줄로 본다 — 첫 쿠션 **칸**(@x,y)은 230 mm 눈금이라 한 칸 옆으로 가는 일이
// 흔하고, 그때 열쇠만 보면 1등에 있는 같은 줄을 44등의 옛 칸에서 찾는다 (첫 실행에서 그랬다).
function rankOf(key, keys) {
  const exact = keys.indexOf(key);
  const near = keys.findIndex((k) => loose(k) === loose(key));
  if (near >= 0) return { rank: near + 1, how: exact === near ? 'same' : 'path' };
  const i = keys.findIndex((k) => family(k) === family(key));
  if (i >= 0) return { rank: i + 1, how: 'route' };
  return { rank: null, how: 'gone' };
}
const GOOD = new Set(['good']);
const BAD = new Set(['bad', 'thin']);   // 아니다 · "가능하지만 프로가 고를 리 없다"

(async () => {
  const browser = await chromium.launch({ executablePath: SHELL, env: { ...process.env, LD_LIBRARY_PATH: LIBS } });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('file://' + (process.env.PAGE || path.join(ROOT, 'build', 'carom_assistant.html')));
  await page.waitForTimeout(1000);

  const out = [];
  for (const r of ledger) {
    const six = ['white', 'yellow', 'red'].map((c) => r.layout[c].join(' ')).join(' ');
    await page.fill('#seed', six);
    await page.press('#seed', 'Enter');
    await page.click(`[data-cue="${r.cue}"]`);
    await page.click('#run');
    await page.waitForFunction(() => !document.body.classList.contains('busy')
      && document.getElementById('run').textContent !== '계산 중…', null, { timeout: 120000 });
    await page.waitForTimeout(300);
    await page.click('#copy');
    await page.waitForTimeout(300);
    const text = await page.evaluate(() => navigator.clipboard.readText());
    const block = /```carom-feedback\n([\s\S]*?)\n```/.exec(text);
    const keys = block ? JSON.parse(block[1]).candidates.map((c) => c.key) : [];
    const then = (r.candidates || []).map((c) => c.key);
    const rows = Object.entries(r.verdicts).map(([key, verdict]) => ({
      key, verdict, before: then.indexOf(key) + 1 || null, ...rankOf(key, keys),
    }));
    const top = keys[0] || null;
    const topVerdict = top ? (r.verdicts[top] || (Object.entries(r.verdicts).find(([k]) => loose(k) === loose(top)) || [])[1] || null) : null;
    out.push({ at: r.at, seed: r.seed || null, rows, top, topVerdict, n: keys.length });
    process.stderr.write('.');
  }
  process.stderr.write('\n');
  await browser.close();

  const goods = out.flatMap((o) => o.rows.filter((x) => GOOD.has(x.verdict)).map((x) => ({ ...x, at: o.at, seed: o.seed })));
  const bads = out.flatMap((o) => o.rows.filter((x) => BAD.has(x.verdict)).map((x) => ({ ...x, at: o.at, seed: o.seed })));
  const inTop = (x, k) => x.rank !== null && x.rank <= k;
  const wasTop = (x, k) => x.before && x.before <= k;
  console.log(`판단한 배치 ${out.length}판 · 맞다 ${goods.length}줄 · 아니다/낮음 ${bads.length}줄 · 페이지 오류 ${errors.length}`);
  console.log(`  맞다 받은 줄   3등 안: 그때 ${goods.filter((x) => wasTop(x, 3)).length} → 지금 ${goods.filter((x) => inTop(x, 3)).length}`
    + ` · 1등: 그때 ${goods.filter((x) => wasTop(x, 1)).length} → 지금 ${goods.filter((x) => inTop(x, 1)).length}`
    + ` · 목록에서 사라짐 ${goods.filter((x) => x.rank === null).length}`);
  console.log(`  아니다 받은 줄 3등 안: 그때 ${bads.filter((x) => wasTop(x, 3)).length} → 지금 ${bads.filter((x) => inTop(x, 3)).length}`);
  const tops = out.map((o) => o.topVerdict);
  console.log(`  지금의 1등이 선수가 판단한 줄: 맞다 ${tops.filter((v) => v === 'good').length} · 아니다/낮음 ${tops.filter((v) => BAD.has(v)).length}`
    + ` · 판단 안 받은 줄 ${tops.filter((v) => !v).length}`);
  const dropped = goods.filter((x) => wasTop(x, 3) && !inTop(x, 3));
  if (dropped.length) {
    console.log('\n  ⚠️ 3등 안에서 밀려난 "맞다" 줄:');
    for (const x of dropped) console.log(`     ${x.at} 씨앗 ${x.seed ?? '-'} · ${x.key} · ${x.before}등 → ${x.rank ?? '없음'}${x.how === 'same' ? '' : ` (${x.how})`}`);
  }
  const rose = bads.filter((x) => !wasTop(x, 3) && inTop(x, 3));
  if (rose.length) {
    console.log('\n  ⚠️ 3등 안으로 올라온 "아니다" 줄:');
    for (const x of rose) console.log(`     ${x.at} 씨앗 ${x.seed ?? '-'} · ${x.key} · ${x.before || '-'}등 → ${x.rank}${x.how === 'same' ? '' : ` (${x.how})`}`);
  }
  if (saveAt) fs.writeFileSync(saveAt, JSON.stringify(out, null, 1));
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
