// photo.js를 합성 사진(tools/synth_photos.py)에 대고 잰다 — 공마다 정답과의 거리 (mm).
//
//   node tools/check_photo.js <사진 폴더>        (폴더에 NNN.png + truth.json)
//
// 테이블은 180도 회전에 대칭이라 인식이 어느 쪽을 (0,0)으로 잡아도 조언은 같다 — 두 쪽 중
// 가까운 것으로 잰다. 거울상은 허용하지 않는다 (당점 좌우가 뒤집히므로 그것은 틀린 것이다).
const fs = require('fs'), path = require('path'), http = require('http');
const HOME = process.env.HOME;
const npx = path.join(HOME, '.npm', '_npx');
const core = fs.readdirSync(npx).map((d) => path.join(npx, d, 'node_modules', 'playwright-core')).find((p) => fs.existsSync(p));
const shellRoot = fs.readdirSync(path.join(HOME, '.cache', 'ms-playwright')).find((d) => d.startsWith('chromium_headless_shell-'));
const { chromium } = require(core);
const dir = path.resolve(process.argv[2]);
const truth = JSON.parse(fs.readFileSync(path.join(dir, 'truth.json'), 'utf8'));
const photoJs = path.join(__dirname, '..', 'src', 'visualization', 'assistant', 'photo.js');
const server = http.createServer((q, r) => {
  const name = decodeURIComponent(q.url.split('?')[0].slice(1));
  const f = name === 'photo.js' ? photoJs : path.join(dir, name);
  if (name === '') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end('<meta charset="utf-8"><script src="photo.js"></script>'); }
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': f.endsWith('.js') ? 'text/javascript' : 'image/png' });
  r.end(fs.readFileSync(f));
}).listen(0);
(async () => {
  const b = await chromium.launch({ executablePath: path.join(HOME, '.cache', 'ms-playwright', shellRoot, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
    env: { ...process.env, LD_LIBRARY_PATH: path.join(HOME, '.cache', 'carom_libs', 'usr', 'lib', 'x86_64-linux-gnu') } });
  const p = await b.newPage();
  p.on('pageerror', (e) => console.log('페이지 오류', e.message));
  await p.goto(`http://localhost:${server.address().port}/`);
  const byPose = {};
  const rows = [];
  for (const [id, t] of Object.entries(truth)) {
    const got = await p.evaluate(async (src) => {
      const im = new Image(); im.src = src; await im.decode();
      const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
      const g = c.getContext('2d'); g.drawImage(im, 0, 0);
      const img = g.getImageData(0, 0, c.width, c.height);
      const t0 = performance.now();
      const res = PHOTO.analyze(img);
      return { ms: performance.now() - t0, balls: res.balls, warnings: res.warnings, camera: res.camera, corners: res.corners };
    }, `/${id}.png`);
    const L = 2844, W = 1422;
    const err = (rot) => Object.entries(t.balls).map(([c, [x, y]]) => {
      const g = got.balls[c];
      if (!g) return Infinity;
      const [gx, gy] = rot ? [L - g.at[0], W - g.at[1]] : g.at;
      return Math.hypot(gx - x, gy - y);
    });
    const e0 = err(false), e1 = err(true);
    const e = Math.max(...e0) <= Math.max(...e1) ? e0 : e1;
    rows.push({ id, pose: t.pose, e, ms: got.ms, warn: got.warnings.length, cam: got.camera });
    (byPose[t.pose] = byPose[t.pose] || []).push(...e);
    console.log(`${id} ${t.pose.padEnd(9)} 오차 ${e.map((v) => (Number.isFinite(v) ? Math.round(v) : '못찾음')).join(' / ').padEnd(16)} mm · ${Math.round(got.ms)} ms`
      + (got.camera ? ` · 카메라 높이 ${Math.round(got.camera.height)} (참 ${t.camera_mm[2]})` : ' · 카메라 추정 없음')
      + (got.warnings.length ? ` · ⚠ ${got.warnings[0].slice(0, 40)}` : ''));
  }
  const med = (v) => { const s = v.filter(Number.isFinite).sort((a, b) => a - b); return s.length ? Math.round(s[s.length >> 1]) : '-'; };
  console.log('\n자세별 — 중앙값 / 최대 / 못 찾음');
  for (const [pose, v] of Object.entries(byPose)) {
    const ok = v.filter(Number.isFinite);
    console.log(`  ${pose.padEnd(9)} ${med(v)} / ${ok.length ? Math.round(Math.max(...ok)) : '-'} mm / ${v.length - ok.length}`);
  }
  const all = rows.flatMap((r) => r.e);
  console.log(`전체 공 ${all.length}개: 중앙값 ${med(all)} mm · 50 mm 안 ${all.filter((v) => v <= 50).length} · 100 mm 안 ${all.filter((v) => v <= 100).length} · 못 찾음 ${all.filter((v) => !Number.isFinite(v)).length}`);
  await b.close(); server.close();
})();
