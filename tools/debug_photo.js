// 사진 한 장에 photo.js의 분석을 겹쳐 그린다 — 변(하늘색), 모서리(빨강), 탁자 윤곽(초록: 쿠션 끝, 노랑: 경기 면),
// 맞춘 다이아몬드(분홍), 다이아몬드 후보(주황 점), 찾은 공(동그라미).
//
//   node tools/debug_photo.js <사진> <출력.png>
const fs = require('fs'), path = require('path'), http = require('http');
const HOME = process.env.HOME;
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const { chromium } = require(core);
const [photo, out] = process.argv.slice(2);
const photoJs = path.join(__dirname, '..', 'src', 'visualization', 'assistant', 'photo.js');
const server = http.createServer((q, r) => {
  const n = q.url.split('?')[0];
  if (n === '/') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end('<meta charset="utf-8"><script src="photo.js"></script>'); }
  const f = n === '/photo.js' ? photoJs : path.resolve(photo);
  r.writeHead(200); r.end(fs.readFileSync(f));
}).listen(0);
(async () => {
  const b = await chromium.launch({ executablePath: path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
    env: { ...process.env, LD_LIBRARY_PATH: path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu') } });
  const p = await b.newPage();
  p.on('pageerror', (e) => console.log('오류', e.stack));
  await p.goto(`http://localhost:${server.address().port}/`);
  const res = await p.evaluate(async () => {
    const im = new Image(); im.src = '/photo'; await im.decode();
    const k = Math.min(1, 1600 / Math.max(im.naturalWidth, im.naturalHeight));
    const c = document.createElement('canvas'); c.width = Math.round(im.naturalWidth * k); c.height = Math.round(im.naturalHeight * k);
    const g = c.getContext('2d'); g.drawImage(im, 0, 0, c.width, c.height);
    const img = g.getImageData(0, 0, c.width, c.height);
    const t0 = performance.now();
    const e = PHOTO.analyzeView(img);
    const ms = performance.now() - t0;
    if (e.error) return { error: e.error, png: c.toDataURL('image/png') };
    const L = 2844, W = 1422, C = 50, s = c.width / 1000;
    g.lineWidth = 2 * s;
    for (const ln of e.lines) { // ax+by+c=0 를 화면 끝까지
      g.strokeStyle = '#0ff'; g.beginPath();
      if (Math.abs(ln[1]) > Math.abs(ln[0])) { g.moveTo(0, -ln[2] / ln[1]); g.lineTo(c.width, -(ln[2] + ln[0] * c.width) / ln[1]); }
      else { g.moveTo(-ln[2] / ln[0], 0); g.lineTo(-(ln[2] + ln[1] * c.height) / ln[0], c.height); }
      g.stroke();
    }
    const poly = (pts, col) => { g.strokeStyle = col; g.lineWidth = 3 * s; g.beginPath(); pts.forEach((q, i) => { const a = PHOTO.apply(e.H, q); i ? g.lineTo(a[0], a[1]) : g.moveTo(a[0], a[1]); }); g.closePath(); g.stroke(); };
    poly([[-C, -C], [L + C, -C], [L + C, W + C], [-C, W + C]], '#0f0');
    poly([[0, 0], [L, 0], [L, W], [0, W]], '#ff0');
    for (const q of e.blobs) { g.fillStyle = '#f80'; g.fillRect(q[0] - 2 * s, q[1] - 2 * s, 4 * s, 4 * s); }
    for (const d of e.diamonds) { g.strokeStyle = '#f0f'; g.lineWidth = 3 * s; g.beginPath(); g.arc(d[0], d[1], 8 * s, 0, 7); g.stroke(); }
    for (const q of e.corners) { g.fillStyle = '#f00'; g.beginPath(); g.arc(q[0], q[1], 8 * s, 0, 7); g.fill(); }
    const col = { white: '#fff', yellow: '#fc0', red: '#f33' };
    for (const [k2, v] of Object.entries(e.balls)) { const a = PHOTO.apply(e.H, v.world); g.strokeStyle = col[k2]; g.lineWidth = 3 * s; g.beginPath(); g.arc(a[0], a[1], 18 * s, 0, 7); g.stroke(); }
    return { ms, score: e.score, diamonds: e.diamonds.length, blobs: e.blobs.length, lines: e.lines.length, warnings: e.warnings,
             balls: Object.fromEntries(Object.entries(e.balls).map(([k2, v]) => [k2, v.world.map(Math.round)])), png: c.toDataURL('image/png') };
  });
  fs.writeFileSync(out, Buffer.from(res.png.split(',')[1], 'base64'));
  delete res.png;
  console.log(JSON.stringify(res));
  await b.close(); server.close();
})();
