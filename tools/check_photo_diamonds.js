// 실제 사진의 틀을 **정답 없이** 잰다 — 레일 다이아몬드 하나를 빼고 나머지로 틀을 맞춘 뒤,
// 뺀 다이아몬드가 탁자 위에서 몇 mm 어긋나는가.
//
// 왜 (2026-10-12, 선수: "사진 재기를 뭐하러 하지. table 위의 diamond의 위치를 분석하면 될텐데"):
// 실제 사진에는 공의 정답이 없어 줄자로 재야 한다고 했는데, 다이아몬드는 자리가 정해져 있다
// (장쿠션 L/8, 단쿠션 W/4 간격). 틀(사진 → 탁자)이 틀리면 공도 틀리므로, 다이아몬드로 틀을 잰다.
// ⚠️ 이것이 재는 것은 **틀**이다. 공 검출(중심 · 높이 밀림)의 오차는 따로다 — 두 장 사진의 공 일치로 본다.
//
//   node tools/check_photo_diamonds.js 사진1.jpg [사진2.jpg ...]
//
// 출력: 사진마다 맞춘 다이아몬드 수, 맞춘 그대로의 잔차, 하나 빼고의 오차 (레일 따라 / 레일 가로질러).
// 가로지르는 오차가 한쪽으로 쏠리면 photo.js의 DIAMOND(쿠션 끝 → 다이아몬드 중심, 70 mm 가정)가 틀린 것이다.
const fs = require('fs'), path = require('path'), http = require('http');
const HOME = process.env.HOME;
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const { chromium } = require(core);
const photos = process.argv.slice(2);
const photoJs = path.join(__dirname, '..', 'src', 'visualization', 'assistant', 'photo.js');
let current = null;
const server = http.createServer((q, r) => {
  const n = q.url.split('?')[0];
  if (n === '/') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end('<meta charset="utf-8"><script src="photo.js"></script>'); }
  const f = n === '/photo.js' ? photoJs : path.resolve(current);
  r.writeHead(200); r.end(fs.readFileSync(f));
}).listen(0);

(async () => {
  const b = await chromium.launch({ executablePath: path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
    env: { ...process.env, LD_LIBRARY_PATH: path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu') } });
  const p = await b.newPage();
  p.on('pageerror', (e) => console.log('오류', e.stack));
  await p.goto(`http://localhost:${server.address().port}/`);
  const all = [];
  for (const photo of photos) {
    current = photo;
    const res = await p.evaluate(async () => {
      const im = new Image(); im.src = '/photo?' + Math.random(); await im.decode();
      const k = Math.min(1, 1600 / Math.max(im.naturalWidth, im.naturalHeight));
      const c = document.createElement('canvas'); c.width = Math.round(im.naturalWidth * k); c.height = Math.round(im.naturalHeight * k);
      const g = c.getContext('2d'); g.drawImage(im, 0, 0, c.width, c.height);
      const e = PHOTO.analyzeView(g.getImageData(0, 0, c.width, c.height));
      if (e.error) return { error: e.error };
      const { L, W } = PHOTO;
      // 탁자 위 오차를 레일 따라 / 레일 가로질러 (바깥이 +)로 나눈다.
      const split = (d, got) => {
        const dx = got[0] - d[0], dy = got[1] - d[1];
        const long = d[1] < 0 || d[1] > W;                    // 장쿠션의 다이아몬드
        const out = long ? (d[1] < 0 ? -1 : 1) : (d[0] < 0 ? -1 : 1);
        return long ? { along: dx, across: dy * out } : { along: dy, across: dx * out };
      };
      const toWorld = (H, q) => PHOTO.apply(PHOTO.inv(H), q);
      const fitted = e.pairs.map(([d, q]) => split(d, toWorld(e.H, q)));
      const loo = [];
      if (e.pairs.length >= 5) {
        e.pairs.forEach(([d, q], i) => {
          const rest = e.pairs.filter((_, j) => j !== i).map(([dd, qq]) => [dd, qq, 1]);
          const H = PHOTO.fitHomography((e.anchorPair ? [e.anchorPair] : []).concat(rest));
          if (H) loo.push(split(d, toWorld(H, q)));
        });
      }
      return { n: e.pairs.length, fitted, loo, balls: Object.fromEntries(Object.entries(e.balls).map(([k2, v]) => [k2, v.world.map(Math.round)])) };
    });
    const name = path.basename(photo);
    if (res.error) { console.log(`${name}: ${res.error}`); continue; }
    const mag = (o) => Math.hypot(o.along, o.across);
    const med = (v) => { const s = [...v].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
    const mean = (v) => v.reduce((a, x) => a + x, 0) / (v.length || 1);
    console.log(`${name}: 다이아몬드 ${res.n}개 · 맞춘 그대로 중앙값 ${med(res.fitted.map(mag)).toFixed(0)} mm`
      + (res.loo.length ? ` · 하나 빼고 중앙값 ${med(res.loo.map(mag)).toFixed(0)} mm, 최대 ${Math.max(...res.loo.map(mag)).toFixed(0)} mm`
        + ` · 가로질러 평균 ${mean(res.loo.map((o) => o.across)).toFixed(0)} mm (바깥 +)` : ' · 하나 빼고는 5개부터')
      + ` · 공 ${JSON.stringify(res.balls)}`);
    all.push(...res.loo);
  }
  if (all.length) {
    const mag = all.map((o) => Math.hypot(o.along, o.across)).sort((x, y) => x - y);
    console.log(`\n전체 하나 빼고 ${all.length}개: 중앙값 ${mag[Math.floor(mag.length / 2)].toFixed(0)} mm · 90% ${mag[Math.floor(mag.length * 0.9)].toFixed(0)} mm`
      + ` · 가로질러 평균 ${(all.reduce((a, o) => a + o.across, 0) / all.length).toFixed(0)} mm`);
  }
  await b.close(); server.close();
})();
