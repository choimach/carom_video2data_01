// 사진 한 장 → 세 공의 테이블 좌표.  **OpenCV 없이 브라우저(와 node)에서 돈다.**
//
// 선수 (2026-10-03, 당구장에서 폰 앱을 써 본 뒤): "공들을 drag해서는 정확한 곳에 위치시키기
// 힘들고, 시간도 많이 걸려. 사진을 찍어서 공 위치를 잡아야겠어."
//
// 형제 저장소 carom_bot_01/src/carom/vision.py (텔레그램 봇, 2026-09)를 옮긴 것이다. 거기서
// 측정해 정한 값들(색 범위, 블롭 크기 허용, 축 배정, 시차 보정)을 그대로 가져왔다 — 근거는
// 그 파일의 주석에 있다. 흐름:
//
//   1) 천(녹색/청색) 영역 → 볼록껍질 → 꼭짓점 넷 (작게 줄인 그림에서)
//   2) 어느 변이 장축인가: 호모그래피 H = K[r1 r2 t]의 두 회전 열은 노름이 같아야 한다
//      (2:1 테이블에서 축을 바꿔 배정하면 2배쯤 어긋난다)
//   3) 위에서 본 그림으로 편다 → 흰·노랑·빨강을 색으로 찾는다
//   4) 카메라 높이를 추정해 **공이 떠 있어서 생기는 겉보기 밀림**을 되돌린다 (높이 1.7 m에서
//      먼 공은 수 cm, 낮게 찍으면 10 cm 넘게)
//
// 자동이 틀릴 수 있으므로 화면은 네 모서리를 손으로 옮길 수 있게 한다 (analyze에 corners를 준다).
//
// 좌표: 돌려주는 공 위치는 조언판과 같은 **mm, x 0~2844, y 0~1422, y는 아래로**.
const PHOTO = (() => {
  const L = 2844, W = 1422, R = 30.75;          // mm
  const PX = 0.5;                                 // 펴진 그림: 1 mm = 0.5 px (vision.py의 500 px/m)
  const PAD = 100;                                // 펴진 그림의 여백 mm — 쿠션에 붙은 공이 잘리지 않게
  const DETECT_SIDE = 480;                        // 테이블 찾기는 이만큼 줄여서
  // ★천 영역의 바깥 테두리는 경기 면의 가장자리가 아니라 **쿠션 윗면의 바깥 끝**이다 (쿠션도
  // 같은 천으로 싸여 있다). 그 선은 경기 면보다 CUSHION mm 바깥, 높이 CUSHION_TOP mm에 있다.
  // 그래서 네 모서리를 그 직사각형에 맞추면 펴진 그림은 **높이 37 mm 평면**이 되고, 공 중심
  // (높이 30.75 mm)은 거의 그 평면 위에 있어 공이 떠 있어서 생기는 밀림이 저절로 사라진다
  // (남는 6 mm 차이는 무시할 만하다). 합성 사진에서 경기 면에 맞췄을 때 오차가 45~90 mm로
  // 한쪽으로 쏠렸다 (2026-10-03). ⚠️ 50 / 37은 대대 쿠션의 대략값 — 실제 사진으로 다시 잴 것.
  const CUSHION = 50, CUSHION_TOP = 37;

  // ── 색 ────────────────────────────────────────────────────────────────
  // OpenCV와 같은 눈금: H 0~179, S 0~255, V 0~255. vision.py의 범위를 그대로 쓰기 위해서다.
  function hsv(r, g, b) {
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    let h = 0;
    if (d > 0) {
      if (max === r) h = 60 * (((g - b) / d) % 6);
      else if (max === g) h = 60 * ((b - r) / d + 2);
      else h = 60 * ((r - g) / d + 4);
      if (h < 0) h += 360;
    }
    // OpenCV처럼 정수로 반올림하고 180은 0으로 — 안 그러면 빨강 끝자락(179.3)이 범위 밖으로 샌다.
    return [Math.round(h / 2) % 180, max === 0 ? 0 : (d / max) * 255, max];
  }
  const COLOR = {
    white: [[[0, 0, 165], [179, 70, 255]]],
    yellow: [[[16, 90, 120], [38, 255, 255]]],
    // 상한 8: 나무 레일(H≈12)이 빨간공으로 잡혔다 (vision.py).
    red: [[[0, 120, 70], [8, 255, 255]], [[172, 120, 70], [179, 255, 255]]],
  };
  const inRange = (p, ranges) => ranges.some(([lo, hi]) =>
    p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1] && p[2] >= lo[2] && p[2] <= hi[2]);

  // ── 그림 다루기 ───────────────────────────────────────────────────────
  // img: {data: Uint8ClampedArray(RGBA), width, height}
  function shrink(img, side) {
    const k = Math.max(1, Math.max(img.width, img.height) / side);
    const w = Math.round(img.width / k), h = Math.round(img.height / k);
    const out = { data: new Uint8ClampedArray(w * h * 4), width: w, height: h, scale: 1 / k };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // 상자 평균 (가까운 이웃이면 공이 사라질 수 있다)
        let r = 0, g = 0, b = 0, n = 0;
        const x0 = Math.floor(x * k), x1 = Math.min(img.width, Math.floor((x + 1) * k));
        const y0 = Math.floor(y * k), y1 = Math.min(img.height, Math.floor((y + 1) * k));
        for (let yy = y0; yy < Math.max(y1, y0 + 1); yy++) {
          for (let xx = x0; xx < Math.max(x1, x0 + 1); xx++) {
            const i = (yy * img.width + xx) * 4;
            r += img.data[i]; g += img.data[i + 1]; b += img.data[i + 2]; n++;
          }
        }
        const o = (y * w + x) * 4;
        out.data[o] = r / n; out.data[o + 1] = g / n; out.data[o + 2] = b / n; out.data[o + 3] = 255;
      }
    }
    return out;
  }

  // 정사각 창 (2r+1)의 팽창/침식 — 가로·세로로 나눠 한다.
  function morph(mask, w, h, r, grow) {
    const pass = (src, horizontal) => {
      const dst = new Uint8Array(src.length);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let v = grow ? 0 : 1;
          for (let d = -r; d <= r; d++) {
            const xx = horizontal ? x + d : x, yy = horizontal ? y : y + d;
            const inside = xx >= 0 && xx < w && yy >= 0 && yy < h;
            const s = inside ? src[yy * w + xx] : 0;
            if (grow ? s : !s) { v = grow ? 1 : 0; break; }
          }
          dst[y * w + x] = v;
        }
      }
      return dst;
    };
    return pass(pass(mask, true), false);
  }
  const close = (m, w, h, r, n = 1) => { for (let i = 0; i < n; i++) m = morph(morph(m, w, h, r, true), w, h, r, false); return m; };
  const open = (m, w, h, r, n = 1) => { for (let i = 0; i < n; i++) m = morph(morph(m, w, h, r, false), w, h, r, true); return m; };

  // 연결 요소 (4-이웃). → [{pixels: [인덱스…]}]
  function components(mask, w, h) {
    const seen = new Uint8Array(mask.length), out = [];
    const stack = [];
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i] || seen[i]) continue;
      const pixels = [];
      stack.push(i); seen[i] = 1;
      while (stack.length) {
        const p = stack.pop();
        pixels.push(p);
        const x = p % w, y = (p - x) / w;
        if (x > 0 && mask[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
        if (x < w - 1 && mask[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
        if (y > 0 && mask[p - w] && !seen[p - w]) { seen[p - w] = 1; stack.push(p - w); }
        if (y < h - 1 && mask[p + w] && !seen[p + w]) { seen[p + w] = 1; stack.push(p + w); }
      }
      out.push({ pixels });
    }
    return out;
  }

  // ── 기하 ──────────────────────────────────────────────────────────────
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  function hull(points) {
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const lower = [], upper = [];
    for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
    for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }
  const polyArea = (p) => Math.abs(p.reduce((s, a, i) => { const b = p[(i + 1) % p.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  // 닫힌 다각형의 더글러스-포이커 (cv2.approxPolyDP의 대역)
  function simplify(poly, eps) {
    const segDist = (p, a, b) => {
      const l = dist(a, b);
      return l < 1e-9 ? dist(p, a) : Math.abs(cross(a, b, p)) / l;
    };
    const dp = (pts) => {
      if (pts.length < 3) return pts;
      let best = 0, at = 0;
      for (let i = 1; i < pts.length - 1; i++) { const d = segDist(pts[i], pts[0], pts[pts.length - 1]); if (d > best) { best = d; at = i; } }
      if (best <= eps) return [pts[0], pts[pts.length - 1]];
      return dp(pts.slice(0, at + 1)).slice(0, -1).concat(dp(pts.slice(at)));
    };
    // 서로 가장 먼 두 점에서 둘로 나눈다
    let a = 0, b = 0, far = -1;
    for (let i = 0; i < poly.length; i++) for (let j = i + 1; j < poly.length; j++) { const d = dist(poly[i], poly[j]); if (d > far) { far = d; a = i; b = j; } }
    const one = poly.slice(a, b + 1), two = poly.slice(b).concat(poly.slice(0, a + 1));
    return dp(one).slice(0, -1).concat(dp(two).slice(0, -1));
  }

  // 잘라내는 삼각형이 가장 작은 꼭짓점부터 지워 넷으로 (vision.py: approxPolyDP의 엡실론을
  // 키우는 방식은 원근이 강할 때 모서리를 1149 px 엉뚱한 데 만들었다).
  function reduceToQuad(pts) {
    pts = pts.map((p) => p.slice());
    while (pts.length > 4) {
      let small = Infinity, at = 0;
      for (let i = 0; i < pts.length; i++) {
        const a = pts[(i - 1 + pts.length) % pts.length], c = pts[i], b = pts[(i + 1) % pts.length];
        const area = Math.abs(cross(a, c, b)) / 2;
        if (area < small) { small = area; at = i; }
      }
      pts.splice(at, 1);
    }
    return pts;
  }

  // 네 점을 늘 같은 회전 방향으로 (무게중심 기준 각도가 줄어드는 쪽). 거울상 배정을
  // 원천적으로 막는다 — 거울상이면 당점 좌우가 뒤집혀 조언 전체가 반대가 된다.
  function orderCyclic(q) {
    const c = [q.reduce((s, p) => s + p[0], 0) / 4, q.reduce((s, p) => s + p[1], 0) / 4];
    return q.slice().sort((a, b) => Math.atan2(b[1] - c[1], b[0] - c[0]) - Math.atan2(a[1] - c[1], a[0] - c[0]));
  }

  // 3×3 행렬 (행 우선 배열 9칸)
  const mul = (A, B) => { const C = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j]; return C; };
  function inv(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
            B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
            C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
  }
  const apply = (H, p) => { const w = H[6] * p[0] + H[7] * p[1] + H[8]; return [(H[0] * p[0] + H[1] * p[1] + H[2]) / w, (H[3] * p[0] + H[4] * p[1] + H[5]) / w]; };
  // 네 점 대응 → 호모그래피 (cv2.getPerspectiveTransform)
  function homography(src, dst) {
    const A = [], bv = [];
    for (let k = 0; k < 4; k++) {
      const [x, y] = src[k], [u, v] = dst[k];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); bv.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); bv.push(v);
    }
    // 가우스 소거
    const n = 8, M = A.map((row, i) => row.concat([bv[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) return null;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const h = M.map((row, i) => row[n] / row[i]);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  // ── 카메라 (vision.py 그대로) ───────────────────────────────────────────
  const line = (p, q) => [p[1] - q[1], q[0] - p[0], p[0] * q[1] - q[0] * p[1]];
  const vanish = (l1, l2) => {
    const v = [l1[1] * l2[2] - l1[2] * l2[1], l1[2] * l2[0] - l1[0] * l2[2], l1[0] * l2[1] - l1[1] * l2[0]];
    return Math.abs(v[2]) < 1e-8 ? null : [v[0] / v[2], v[1] / v[2]];
  };
  const VANISH_FAR = 8.0, FOCAL_PRIOR = 0.72, MAX_PARALLAX = 130, LOW_CAMERA = 1350;
  const isFar = (v, c0, diag) => v === null || dist(v, c0) > VANISH_FAR * diag;
  // mm, y는 위로 (vision.py와 같다). 사진의 네 모서리가 가리키는 직사각형 = 쿠션 바깥 끝.
  const WORLD = [[-CUSHION, -CUSHION], [L + CUSHION, -CUSHION], [L + CUSHION, W + CUSHION], [-CUSHION, W + CUSHION]];
  function columns(f, c0, corners) {
    const K = [f, 0, c0[0], 0, f, c0[1], 0, 0, 1];
    const H = homography(WORLD, corners);
    if (!H) return null;
    const M = mul(inv(K), H);
    return M;
  }
  const col = (M, j) => [M[j], M[3 + j], M[6 + j]];
  const norm3 = (v) => Math.hypot(v[0], v[1], v[2]);
  function columnRatio(f, c0, corners) {
    const M = columns(f, c0, corners);
    if (!M) return null;
    const n1 = norm3(col(M, 0)), n2 = norm3(col(M, 1));
    return n1 < 1e-12 || n2 < 1e-12 ? null : Math.log(n1 / n2);
  }
  function focalFromVanishing(vx, vy, c0, diag) {
    if (isFar(vx, c0, diag) || isFar(vy, c0, diag)) return null;
    const f2 = -((vx[0] - c0[0]) * (vy[0] - c0[0]) + (vx[1] - c0[1]) * (vy[1] - c0[1]));
    return Number.isFinite(f2) && f2 > 1 ? Math.sqrt(f2) : null;
  }
  function focalFromAspect(corners, c0, wImg) {
    let lo = 0.25 * wImg, hi = 4 * wImg;
    let fLo = columnRatio(lo, c0, corners);
    const fHi = columnRatio(hi, c0, corners);
    if (fLo === null || fHi === null || fLo * fHi > 0) return null;
    for (let i = 0; i < 60; i++) {
      const mid = Math.sqrt(lo * hi), v = columnRatio(mid, c0, corners);
      if (v === null) return null;
      if (v * fLo > 0) { lo = mid; fLo = v; } else hi = mid;
    }
    return Math.sqrt(lo * hi);
  }
  function cameraFromFocal(f, c0, corners) {
    const M = columns(f, c0, corners);
    if (!M) return null;
    const n1 = norm3(col(M, 0));
    if (n1 < 1e-12) return null;
    for (const sign of [1, -1]) {
      const lam = sign / n1;
      const r1 = col(M, 0).map((v) => v * lam), r2 = col(M, 1).map((v) => v * lam), t = col(M, 2).map((v) => v * lam);
      const r3 = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]];
      // cam = −Rᵀ t, R = [r1 r2 r3] (열)
      const cam = [-(r1[0] * t[0] + r1[1] * t[1] + r1[2] * t[2]),
                   -(r2[0] * t[0] + r2[1] * t[1] + r2[2] * t[2]),
                   -(r3[0] * t[0] + r3[1] * t[1] + r3[2] * t[2])];
      if (cam[2] > 0) return cam;
    }
    return null;
  }
  // 카메라의 지면 투영점(nadir, 테이블 mm, y 위로)과 높이(mm). 원근이 없으면 null.
  function estimateCamera(corners, wImg, hImg) {
    const [A, B, C, D] = corners;
    const vx = vanish(line(A, B), line(D, C)), vy = vanish(line(A, D), line(B, C));
    const c0 = [wImg / 2, hImg / 2], diag = Math.hypot(wImg, hImg);
    if (isFar(vx, c0, diag) && isFar(vy, c0, diag)) return null;
    const tries = [];
    let f = focalFromVanishing(vx, vy, c0, diag);
    if (f !== null && f > 0.3 * wImg && f < 4 * wImg) tries.push(f);
    f = focalFromAspect(corners, c0, wImg);
    if (f !== null && f > 0.3 * wImg && f < 4 * wImg) tries.push(f);
    tries.push(FOCAL_PRIOR * wImg);
    for (const ff of tries) {
      const cam = cameraFromFocal(ff, c0, corners);
      if (cam && cam[2] > 500 && cam[2] < 4000) return { nadir: [cam[0], cam[1]], height: cam[2] };
    }
    return null;
  }
  // 겉보기 중심 → 접지 위치: nadir 쪽으로 (1 − R/h)배 당긴다.
  function correctParallax(xy, cam) {
    if (!cam || cam.height <= R) return xy;
    let sx = (R / cam.height) * (xy[0] - cam.nadir[0]), sy = (R / cam.height) * (xy[1] - cam.nadir[1]);
    const mag = Math.hypot(sx, sy);
    if (mag > MAX_PARALLAX) { sx *= MAX_PARALLAX / mag; sy *= MAX_PARALLAX / mag; }
    return [xy[0] - sx, xy[1] - sy];
  }
  // 이 축 배정이 카메라 기하와 얼마나 모순되나 (0에 가까울수록 옳다). 아핀이면 null.
  function aspectConsistency(corners, wImg, hImg) {
    const [A, B, C, D] = corners;
    const c0 = [wImg / 2, hImg / 2], diag = Math.hypot(wImg, hImg);
    const vx = vanish(line(A, B), line(D, C)), vy = vanish(line(A, D), line(B, C));
    if (isFar(vx, c0, diag) && isFar(vy, c0, diag)) return null;
    let f = focalFromVanishing(vx, vy, c0, diag);
    if (f === null || !(f > 0.3 * wImg && f < 4 * wImg)) f = FOCAL_PRIOR * wImg;
    const M = columns(f, c0, corners);
    if (!M) return null;
    const n1 = norm3(col(M, 0)), n2 = norm3(col(M, 1));
    return n1 < 1e-12 || n2 < 1e-12 ? null : Math.abs(Math.log(n1 / n2));
  }

  // ── 테이블 찾기 ───────────────────────────────────────────────────────
  // 천 영역 (줄인 그림에서): {small, w, h, big(가장 큰 연결 요소), edge(줄마다 양 끝 점)} 또는 {error}
  function clothRegion(img) {
    const small = shrink(img, DETECT_SIDE);
    const { width: w, height: h, data } = small;
    const px = new Array(w * h);
    let solid = 0;
    for (let i = 0; i < w * h; i++) { px[i] = hsv(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]); if (px[i][1] > 70 && px[i][2] > 45) solid++; }
    const loose = solid < 0.05 * w * h;
    const hist = new Array(180).fill(0);
    for (const p of px) if (loose ? (p[1] > 40 && p[2] > 30) : (p[1] > 70 && p[2] > 45)) hist[Math.min(179, Math.floor(p[0]))]++;
    // 녹색(35~95)과 청색(95~130)만 — 나무 레일과 살색을 뺀다
    let hue = 35;
    for (let k = 35; k < 130; k++) if (hist[k] > hist[hue]) hue = k;
    const tol = 18;
    let mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) { const p = px[i]; mask[i] = p[0] >= hue - tol && p[0] <= hue + tol && p[1] >= 55 && p[2] >= 40 ? 1 : 0; }
    const r = Math.max(1, Math.round(4 * w / 1600));      // 9×9 창을 줄인 그림 크기에 맞춘다
    mask = open(close(mask, w, h, r, 3), w, h, r, 1);
    const comps = components(mask, w, h);
    if (!comps.length) return { error: "천(라사)을 찾지 못했습니다. 다시 찍어 주세요." };
    const big = comps.reduce((a, b) => (b.pixels.length > a.pixels.length ? b : a));
    if (big.pixels.length < 0.10 * w * h) return { error: "천 영역이 너무 작습니다. 테이블이 화면을 채우게 찍어 주세요." };
    // 줄마다 양 끝만 모아 껍질을 만든다 (공이 낸 구멍은 껍질이 메운다)
    const rows = new Map();
    for (const p of big.pixels) {
      const x = p % w, y = (p - x) / w;
      const row = rows.get(y);
      if (!row) rows.set(y, [x, x]); else { if (x < row[0]) row[0] = x; if (x > row[1]) row[1] = x; }
    }
    const edge = [];
    for (const [y, [a, b]] of rows) { edge.push([a, y], [b + 1, y], [a, y + 1], [b + 1, y + 1]); }
    // 칸마다 위아래 끝도 (2026-10-04: 거의 가로로 누운 변은 줄 끝 점만으로는 양 끝 두 점뿐이었다)
    const cols = new Map();
    for (const p of big.pixels) {
      const x = p % w, y = (p - x) / w;
      const c = cols.get(x);
      if (!c) cols.set(x, [y, y]); else { if (y < c[0]) c[0] = y; if (y > c[1]) c[1] = y; }
    }
    const sides = [];
    for (const [x, [a, b]] of cols) sides.push([x + 0.5, a], [x + 0.5, b + 1]);
    return { small, w, h, big, edge, sides, mask };
  }

  function findTable(img) {
    const region = clothRegion(img);
    if (region.error) return { corners: null, confidence: 0, warnings: [region.error] };
    const { small, w, h, big, edge } = region;
    const H0 = hull(edge);
    const peri = H0.reduce((s, p, i) => s + dist(p, H0[(i + 1) % H0.length]), 0);
    const approx = simplify(H0, 0.012 * peri);
    let quad, conf;
    if (approx.length < 4) { quad = reduceToQuad(H0); conf = 0.5; }
    else { quad = reduceToQuad(approx); conf = approx.length <= 6 ? 0.9 : 0.75; }
    conf *= Math.min(1, big.pixels.length / Math.max(polyArea(quad), 1e-6));
    const warnings = [];
    const m = 3;
    if (quad.some(([x, y]) => x <= m || y <= m || x >= w - 1 - m || y >= h - 1 - m)) {
      conf *= 0.35;
      warnings.push("테이블이 사진 밖으로 잘린 것 같습니다. 네 모서리가 모두 보이게 한 걸음 물러서 찍거나, 모서리를 직접 옮겨 주세요.");
    }
    if (conf < 0.6) warnings.push("테이블 모서리가 불확실합니다. 모서리 점이 천의 네 귀퉁이에 맞는지 확인해 주세요.");
    const back = 1 / small.scale;
    return { corners: orderCyclic(quad.map(([x, y]) => [x * back, y * back])), confidence: conf, warnings };
  }

  // 순환 정렬된 넷 중 어느 것이 (0,0)인가. ① 장축/단축은 카메라 기하로 ② 180도는 가까운 변을 y=0.
  function pickAssignment(cyclic, wImg, hImg) {
    const rots = [0, 1, 2, 3].map((k) => cyclic.slice(k).concat(cyclic.slice(0, k)));
    const cons = rots.map((r) => aspectConsistency(r, wImg, hImg));
    let pair;
    if (cons.every((c) => c !== null)) pair = Math.min(cons[0], cons[2]) <= Math.min(cons[1], cons[3]) ? [0, 2] : [1, 3];
    else {
      const axis = (r) => (dist(r[1], r[0]) + dist(r[2], r[3])) / 2;
      pair = axis(rots[0]) >= axis(rots[1]) ? [0, 2] : [1, 3];
    }
    const k = dist(rots[pair[0]][1], rots[pair[0]][0]) >= dist(rots[pair[1]][1], rots[pair[1]][0]) ? pair[0] : pair[1];
    return rots[k];
  }

  // ── 펴기와 공 찾기 ─────────────────────────────────────────────────────
  const TOP_W = Math.round((L + 2 * PAD) * PX), TOP_H = Math.round((W + 2 * PAD) * PX);
  // 모서리 (0,0),(L,0),(L,W),(0,W) [y 위로] → 펴진 그림 픽셀 [y 아래로]
  const topCorners = () => WORLD.map(([x, y]) => [(x + PAD) * PX, (W - y + PAD) * PX]);
  function warp(img, corners) { return warpWith(img, homography(topCorners(), corners)); }
  // 테이블 좌표(mm, y 위로) → 사진 호모그래피로 편다
  const TOP_TO_WORLD = [1 / PX, 0, -PAD, 0, -1 / PX, W + PAD, 0, 0, 1];
  function warpWorld(img, Hw2i) { return warpWith(img, mul(Hw2i, TOP_TO_WORLD)); }
  function warpWith(img, H) {                                  // H: 펴진 그림 → 사진
    const out = { data: new Uint8ClampedArray(TOP_W * TOP_H * 4), width: TOP_W, height: TOP_H };
    const { width: sw, height: sh, data } = img;
    for (let y = 0; y < TOP_H; y++) {
      for (let x = 0; x < TOP_W; x++) {
        const [u, v] = apply(H, [x + 0.5, y + 0.5]);
        const o = (y * TOP_W + x) * 4;
        if (!(u >= 0 && v >= 0 && u < sw - 1 && v < sh - 1)) { out.data[o + 3] = 255; continue; }
        const x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0;
        for (let c = 0; c < 3; c++) {
          const i = (y0 * sw + x0) * 4 + c;
          out.data[o + c] = (data[i] * (1 - fx) + data[i + 4] * fx) * (1 - fy)
            + (data[i + sw * 4] * (1 - fx) + data[i + sw * 4 + 4] * fx) * fy;
        }
        out.data[o + 3] = 255;
      }
    }
    return out;
  }
  // 펴진 그림 픽셀 → 테이블 mm (y 위로; vision.py의 px_to_table)
  const topToWorld = (p) => [p[0] / PX - PAD, W - (p[1] / PX - PAD)];

  // 가장 작은 둘러싼 원 (웰즐, 반복형)
  function enclosing(points) {
    const circle2 = (a, b) => ({ c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], r: dist(a, b) / 2 });
    const circle3 = (a, b, c) => {
      const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
      if (Math.abs(d) < 1e-12) return null;
      const a2 = a[0] ** 2 + a[1] ** 2, b2 = b[0] ** 2 + b[1] ** 2, c2 = c[0] ** 2 + c[1] ** 2;
      const ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
      const uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
      return { c: [ux, uy], r: dist([ux, uy], a) };
    };
    const inside = (k, p) => k && dist(k.c, p) <= k.r + 1e-7;
    const pts = points.slice();
    for (let i = pts.length - 1; i > 0; i--) { const j = (i * 7919) % (i + 1); [pts[i], pts[j]] = [pts[j], pts[i]]; }
    let k = { c: pts[0], r: 0 };
    for (let i = 1; i < pts.length; i++) {
      if (inside(k, pts[i])) continue;
      k = { c: pts[i], r: 0 };
      for (let j = 0; j < i; j++) {
        if (inside(k, pts[j])) continue;
        k = circle2(pts[i], pts[j]);
        for (let m = 0; m < j; m++) {
          if (inside(k, pts[m])) continue;
          k = circle3(pts[i], pts[j], pts[m]) || k;
        }
      }
    }
    return k;
  }

  // 블롭 크기·모양 허용 — 측정해서 정한 값 (vision.py: 합성 사진 63건에서 면적비 0.39~4.18).
  const AREA_MIN = 0.30, AREA_MAX = 6.0, MIN_CIRC = 0.25, R_OK = [0.9, 2.9];
  function detectBalls(top) {
    const { width: w, height: h } = top;
    // 3×3 평균을 두 번 (GaussianBlur 5×5의 대역)
    let src = top.data;
    for (let pass = 0; pass < 2; pass++) {
      const dst = new Uint8ClampedArray(src.length);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
        let s = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) { s += src[(yy * w + xx) * 4 + c]; n++; }
        }
        dst[(y * w + x) * 4 + c] = s / n;
      }
      src = dst;
    }
    const px = new Array(w * h);
    for (let i = 0; i < w * h; i++) px[i] = hsv(src[i * 4], src[i * 4 + 1], src[i * 4 + 2]);
    const expectR = R * PX, expectArea = Math.PI * expectR * expectR;
    const found = {}, warnings = [];
    for (const colour of ["white", "yellow", "red"]) {
      let mask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) mask[i] = inRange(px[i], COLOR[colour]) ? 1 : 0;
      mask = close(open(mask, w, h, 2), w, h, 2);
      let best = null;
      for (const comp of components(mask, w, h)) {
        const area = comp.pixels.length;
        if (area < AREA_MIN * expectArea || area > AREA_MAX * expectArea) continue;
        const set = new Set(comp.pixels), border = [];
        for (const p of comp.pixels) {
          const x = p % w, y = (p - x) / w;
          if (!set.has(p - 1) || !set.has(p + 1) || !set.has(p - w) || !set.has(p + w)) border.push([x + 0.5, y + 0.5]);
        }
        const k = enclosing(border);
        const r = Math.max(k.r, 0.5);
        // 공 중심은 쿠션 안쪽에만 있을 수 있다 (2026-10-04, 실제 사진: 쿠션의 반사광을 흰공으로 잡았다,
        // x = −65 mm). 겉보기 밀림을 생각해 공 반지름만큼은 봐준다.
        const wpos = topToWorld(k.c);
        if (wpos[0] < -R || wpos[0] > L + R || wpos[1] < -R || wpos[1] > W + R) continue;
        const circ = area / (Math.PI * r * r);
        if (circ < MIN_CIRC) continue;
        const ratio = r / expectR;
        const excess = Math.max(0, R_OK[0] - ratio) / 0.3 + Math.max(0, ratio - R_OK[1]) / 0.8;
        const score = 0.55 * Math.min(1, circ) + 0.45 * Math.exp(-excess);
        if (!best || score > best.score) best = { c: k.c, r, score };
      }
      if (!best) { warnings.push(`${{ white: "흰공", yellow: "노란공", red: "빨간공" }[colour]}을 찾지 못했습니다.`); continue; }
      found[colour] = { world: topToWorld(best.c), confidence: Math.max(0, Math.min(1, best.score)), top: best.c };
    }
    const ids = Object.keys(found);
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const a = found[ids[i]], b = found[ids[j]];
      if (dist(a.world, b.world) < 1.5 * R) {
        const loser = a.confidence < b.confidence ? ids[i] : ids[j];
        found[loser].confidence = Math.min(found[loser].confidence, 0.2);
        warnings.push("두 공이 같은 자리로 잡혔습니다. 확인해 주세요.");
      }
    }
    return { found, warnings };
  }

  // ── 진입점 ────────────────────────────────────────────────────────────
  // img: RGBA (이미 1600 px 안쪽으로 줄인 것). corners를 주면 그 넷을 (0,0),(L,0),(L,W),(0,W)로 쓴다.
  // → { corners, balls: {colour: {at: [x, y] (조언판 mm, y 아래로), confidence}}, warnings, camera, top }
  function analyze(img, corners = null) {
    const warnings = [];
    let chosen = corners, confidence = 1;
    if (!chosen) {
      const t = findTable(img);
      warnings.push(...t.warnings);
      if (!t.corners) return { corners: null, balls: {}, warnings, camera: null, top: null };
      chosen = pickAssignment(t.corners, img.width, img.height);
      confidence = t.confidence;
    }
    const top = warp(img, chosen);
    const { found, warnings: w2 } = detectBalls(top);
    warnings.push(...w2);
    const camera = estimateCamera(chosen, img.width, img.height);
    if (camera && camera.height < LOW_CAMERA) {
      warnings.push(`카메라가 낮습니다 (약 ${(camera.height / 1000).toFixed(1)} m). 조금 높이 들고 내려다보듯 찍으면 훨씬 정확합니다.`);
      for (const b of Object.values(found)) b.confidence = Math.min(b.confidence, 0.5);
    }
    const balls = {};
    for (const [colour, b] of Object.entries(found)) {
      // 펴진 평면은 쿠션 윗면 높이다. 공 중심은 그보다 (CUSHION_TOP − R)만큼 **아래**라 nadir에서
      // 멀어지는 쪽으로 아주 조금 밀려 있다 — 그만큼만 되돌린다 (높이 1.6 m, 거리 2 m에서 8 mm).
      const [x, y] = camera ? (() => {
        const k = (CUSHION_TOP - R) / (camera.height - CUSHION_TOP);
        return [b.world[0] + k * (b.world[0] - camera.nadir[0]), b.world[1] + k * (b.world[1] - camera.nadir[1])];
      })() : b.world;
      balls[colour] = { at: [Math.max(R, Math.min(L - R, x)), Math.max(R, Math.min(W - R, W - y))],
                        confidence: b.confidence };
    }
    return { corners: chosen, confidence, balls, warnings, camera, top };
  }

  // ── 두 장으로 나눠 찍기 (2026-10-03) ──────────────────────────────────────
  // 선수: "한번에 테이블 전체가 나오게 찍는 건 불가능해. table을 부분적으로 두 번 찍은 후 공들의
  // 위치, 코너와 다이아몬드 포인트의 위치 등을 분석한 후 table에 공들을 위치시키는 방법을 찾아야 해."
  //
  // 사진 한 장 = 탁자의 한쪽 끝 (단쿠션 하나와 그 두 모서리가 보이고 나머지는 잘림). 그 끝을
  // **자기 틀**의 x = −CUSHION 쪽 끝으로 놓고 푼다. 두 번째 장은 반대쪽 끝이므로 180도 돌려 합친다.
  //
  //   1) 천 테두리에서 화면 가장자리에 걸린 변을 버리면 "장쿠션 – 단쿠션 – 장쿠션"이 남는다.
  //      두 모서리 = 세 변의 교점.
  //   2) 처음 추정: 두 모서리 + 장쿠션 소실점 + 화각 사전값(직교 조건으로 단축 소실점을 정한다).
  //   3) 다이아몬드가 있어야 할 자리를 예측하고, 찾은 밝은 점과 짝지어 최소제곱으로 다듬는다 (세 번).
  //      다이아몬드: 장축 L/8, 단축 W/4 간격, 쿠션 바깥 끝에서 DIAMOND mm 바깥.
  const DIAMOND = 70;          // ⚠️ 쿠션 바깥 끝 → 다이아몬드 중심 (대략 레일 폭의 절반). 실제 사진으로 잴 것.
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const h3 = (p) => [p[0], p[1], 1];
  const mv = (M, v) => [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[3] * v[0] + M[4] * v[1] + M[5] * v[2], M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];

  // 점들에 직선 맞추기 (전최소제곱). → [a, b, c] (ax + by + c = 0, a²+b²=1)
  function fitLine(points) {
    const n = points.length, mx = points.reduce((s, p) => s + p[0], 0) / n, my = points.reduce((s, p) => s + p[1], 0) / n;
    let sxx = 0, sxy = 0, syy = 0;
    for (const [x, y] of points) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); syy += (y - my) ** 2; }
    const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);           // 방향
    const a = -Math.sin(ang), b = Math.cos(ang);
    return [a, b, -(a * mx + b * my)];
  }
  // 열린 사슬을 꼭짓점 넷으로 (양 끝은 둔다)
  function reduceChain(pts, keep = 4) {
    pts = pts.map((p) => p.slice());
    while (pts.length > keep) {
      let small = Infinity, at = 1;
      for (let i = 1; i < pts.length - 1; i++) {
        const area = Math.abs(cross(pts[i - 1], pts[i], pts[i + 1])) / 2;
        if (area < small) { small = area; at = i; }
      }
      pts.splice(at, 1);
    }
    return pts;
  }
  // 가중 최소제곱 DLT: [[world(mm, y 위로), image(px), 무게]…] → 테이블 → 사진 호모그래피
  function fitHomography(pairs) {
    const sw = 1 / 1000, si = 1 / 1000;                        // 숫자 크기를 맞춘다
    const N = Array.from({ length: 8 }, () => new Array(9).fill(0));
    for (const [[X0, Y0], [u0, v0], wt] of pairs) {
      const X = X0 * sw, Y = Y0 * sw, u = u0 * si, v = v0 * si;
      for (const [row, rhs] of [[[X, Y, 1, 0, 0, 0, -u * X, -u * Y], u], [[0, 0, 0, X, Y, 1, -v * X, -v * Y], v]]) {
        for (let i = 0; i < 8; i++) { for (let j = 0; j < 8; j++) N[i][j] += wt * row[i] * row[j]; N[i][8] += wt * row[i] * rhs; }
      }
    }
    for (let c = 0; c < 8; c++) {
      let piv = c;
      for (let r = c + 1; r < 8; r++) if (Math.abs(N[r][c]) > Math.abs(N[piv][c])) piv = r;
      [N[c], N[piv]] = [N[piv], N[c]];
      if (Math.abs(N[c][c]) < 1e-14) return null;
      for (let r = 0; r < 8; r++) { if (r === c) continue; const f = N[r][c] / N[c][c]; for (let k = c; k < 9; k++) N[r][k] -= f * N[c][k]; }
    }
    const h = N.map((row, i) => row[8] / row[i]);
    const Hn = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
    // 정규화를 되돌린다: H = S_i⁻¹ · Hn · S_w
    return mul(mul([1 / si, 0, 0, 0, 1 / si, 0, 0, 0, 1], Hn), [sw, 0, 0, 0, sw, 0, 0, 0, 1]);
  }
  // 이 끝에서 보일 수 있는 다이아몬드 (자기 틀, y 위로)
  const DIAMONDS = (() => {          // 탁자 전체 20개 (세계 mm, y 위로)
    const out = [];
    for (let k = 1; k <= 7; k++) { out.push([k * L / 8, -CUSHION - DIAMOND]); out.push([k * L / 8, W + CUSHION + DIAMOND]); }
    for (let j = 1; j <= 3; j++) { out.push([-CUSHION - DIAMOND, j * W / 4]); out.push([L + CUSHION + DIAMOND, j * W / 4]); }
    return out;
  })();
  // 밝고 채도 낮은 작은 점 (천 밖) — 다이아몬드 후보
  function diamondBlobs(img, clothMaskFull) {
    const { width: w, height: h, data } = img;
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (clothMaskFull(i % w, (i - (i % w)) / w)) continue;
      const p = hsv(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
      // 채도 60 미만: 실제 다이아몬드는 5~22, 밝은 레일은 90~110이라 110으로는 레일과 붙었다 (2026-10-04)
      mask[i] = p[2] > 165 && p[1] < 60 ? 1 : 0;
    }
    const out = [];
    for (const c of components(mask, w, h)) {
      const n = c.pixels.length;
      if (n < 3 || n > 900) continue;
      let sx = 0, sy = 0;
      for (const p of c.pixels) { sx += p % w; sy += (p - (p % w)) / w; }
      const cx = sx / n + 0.5, cy = sy / n + 0.5;
      // 길쭉하면 다이아몬드가 아니라 레일 모서리를 따라 생긴 반사광 줄이다 (2026-10-04, 실제 사진).
      // 다이아몬드는 비스듬히 보여도 납작한 타원 정도 — 긴 축이 짧은 축의 3배를 넘지 않는다.
      let xx = 0, yy = 0, xy = 0;
      for (const p of c.pixels) { const dx = p % w + 0.5 - cx, dy = (p - (p % w)) / w + 0.5 - cy; xx += dx * dx; yy += dy * dy; xy += dx * dy; }
      xx /= n; yy /= n; xy /= n;
      const tr = xx + yy, det = xx * yy - xy * xy, gap = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const major = tr / 2 + gap, minor = Math.max(1e-6, tr / 2 - gap);
      if (n >= 6 && major / minor > 9) continue;
      // ★둘레가 나무여야 한다 (2026-10-04, 실제 사진: 바닥 타일의 밝은 무늬가 후보로 넘쳤다). 점 둘레 여덟
      // 군데 중 여섯 이상이 레일 색(갈색: H 4~28, 채도·밝기 있음)이어야 다이아몬드로 본다.
      const r = Math.max(4, 2.2 * Math.sqrt(n / Math.PI));
      let wood = 0;
      for (let k = 0; k < 8; k++) {
        const x = Math.round(cx + r * Math.cos(k * Math.PI / 4)), y = Math.round(cy + r * Math.sin(k * Math.PI / 4));
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const i = (y * w + x) * 4, q = hsv(data[i], data[i + 1], data[i + 2]);
        if (q[0] >= 4 && q[0] <= 28 && q[1] > 70 && q[2] > 50) wood++;
      }
      if (wood >= 6) out.push([cx, cy]);
    }
    return out;
  }

  // 사진 한 장 → 탁자 좌표로 가는 호모그래피와 공 자리. **보이는 만큼만 있으면 된다.**
  //
  // ★2026-10-04, 선수의 실제 사진 두 장: 탁자 모서리에 서서 대각선으로 찍어 **모서리는 바로 앞의
  // 하나만** 보이고, 거기서 뻗는 두 레일과 맞은편 레일 일부가 보였다. 처음 짠 "단쿠션 + 두 모서리"
  // 가정으로는 못 읽는다. 그래서 일반형으로:
  //   · 천 테두리에서 화면 가장자리 구간을 빼면 보이는 변들이 남는다 (끊긴 여러 토막일 수 있다).
  //   · 가설 = (기준 모서리, 거기서 뻗는 두 변 중 어느 것이 장쿠션인가, 평행한 맞은편 변) —
  //     맞은편 변이 소실점 하나와 축척(두 레일 사이 거리)을 준다. 다른 축의 소실점은 화각 f에서
  //     직교 조건으로. f는 훑는다.
  //   · 기준 모서리를 네 모서리 중 어디에 둘지 다 놓아 보고, **위에서 본 거울상이 아닌 것**만 남긴다.
  //   · 점수 = 레일 다이아몬드 20개의 예측 자리와 찾은 밝은 점이 겹치는 정도 → 최소제곱으로 다듬기.
  // 같은 점수의 180도 쌍둥이는 구별할 수 없다 — 두 장을 합칠 때 공이 겹치는 쪽으로 정한다.
  function analyzeView(img, opts = {}) {
    const warnings = [];
    const region = clothRegion(img);
    if (region.error) return { error: region.error };
    const back = 1 / region.small.scale;
    const { w, h } = region;
    const edge = region.edge.concat(region.sides);
    const H0 = hull(region.edge);
    const m = 2.5;
    const onBorder = (p) => p[0] <= m || p[1] <= m || p[0] >= w - m || p[1] >= h - m;
    const sameSide = (a, b) => (a[0] <= m && b[0] <= m) || (a[1] <= m && b[1] <= m)
      || (a[0] >= w - m && b[0] >= w - m) || (a[1] >= h - m && b[1] >= h - m);
    // 껍질을 화면 가장자리 구간에서 끊어 토막들로
    const n = H0.length;
    const isBorderEdge = (i) => onBorder(H0[i]) && onBorder(H0[(i + 1) % n]) && sameSide(H0[i], H0[(i + 1) % n]);
    let start = 0;
    while (start < n && !isBorderEdge(start)) start++;
    const runs = [];
    if (start === n) runs.push(H0.concat([H0[0]]));          // 가장자리에 안 걸림 = 탁자 전체
    else {
      let cur = null;
      for (let k = 1; k <= n; k++) {
        const i = (start + k) % n;
        if (isBorderEdge((i - 1 + n) % n) || cur === null) { if (cur && cur.length > 1) runs.push(cur); cur = [H0[i]]; }
        if (isBorderEdge(i)) { if (cur.length > 1) runs.push(cur); cur = null; continue; }
        cur.push(H0[(i + 1) % n]);
      }
      if (cur && cur.length > 1) runs.push(cur);
    }
    // 토막마다 꺾은선으로 → 변들과 모서리들
    const peri = H0.reduce((s, p, i) => s + dist(p, H0[(i + 1) % n]), 0);
    const segDist = (p, a, b) => { const l = dist(a, b); return l < 1e-9 ? dist(p, a) : Math.abs(cross(a, b, p)) / l; };
    const within = (p, a, b) => { const t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (dist(a, b) ** 2 || 1); return t > 0.06 && t < 0.94; };
    const lineOf = (a, b) => {
      const near = edge.filter((p) => !onBorder(p) && segDist(p, a, b) < 2.5 && within(p, a, b));
      const l = near.length >= 6 ? fitLine(near) : fitLine([a, b]);
      return [l[0], l[1], l[2] * back];
    };
    const lines = [], corners = [];
    for (const run of runs) {
      const dp = (pts, eps) => {
        if (pts.length < 3) return pts;
        let best = 0, at = 0;
        for (let i = 1; i < pts.length - 1; i++) { const d = segDist(pts[i], pts[0], pts[pts.length - 1]); if (d > best) { best = d; at = i; } }
        return best <= eps ? [pts[0], pts[pts.length - 1]] : dp(pts.slice(0, at + 1), eps).slice(0, -1).concat(dp(pts.slice(at), eps));
      };
      const poly = dp(run, 0.012 * peri).filter((p, i, arr) => i === 0 || dist(p, arr[i - 1]) > 1);
      const first = lines.length;
      for (let i = 0; i + 1 < poly.length; i++) {
        if (dist(poly[i], poly[i + 1]) < 0.04 * Math.max(w, h)) continue;    // 짧은 토막은 버린다
        lines.push({ l: lineOf(poly[i], poly[i + 1]), a: poly[i], b: poly[i + 1], run: runs.indexOf(run) });
      }
      for (let i = first; i + 1 < lines.length; i++) {
        if (lines[i].run !== lines[i + 1].run) continue;
        const v = cross3(lines[i].l, lines[i + 1].l);
        if (Math.abs(v[2]) < 1e-9) continue;
        corners.push({ at: [v[0] / v[2], v[1] / v[2]], p: i, q: i + 1 });
      }
    }
    if (!corners.length) return { error: "탁자 모서리가 하나도 안 보입니다. 모서리 하나와 거기서 뻗는 두 레일이 나오게 찍어 주세요." };
    // 다이아몬드 후보와 짝짓기
    const clothSmall = region.mask;
    const inCloth = (x, y) => {
      const sx = Math.min(w - 1, Math.max(0, Math.round(x / back))), sy = Math.min(h - 1, Math.max(0, Math.round(y / back)));
      return clothSmall[sy * w + sx] === 1;
    };
    // ★다이아몬드는 레일 위에만 있다 — 천 테두리 바로 바깥 띠 안의 점만 후보로 (2026-10-04: 실제 사진에서
    // 바닥 타일·벽·조명의 밝은 점까지 1,125개가 후보가 되어 엉뚱한 가설이 이겼다).
    const band = (() => {
      const reach = Math.max(3, Math.round(0.07 * Math.max(w, h)));     // 띠 폭 (줄인 그림 픽셀)
      const d = new Int16Array(w * h).fill(-1);
      const queue = [];
      for (let i = 0; i < w * h; i++) if (clothSmall[i]) { d[i] = 0; queue.push(i); }
      for (let qi = 0; qi < queue.length; qi++) {
        const i = queue[qi], x = i % w, y = (i - x) / w;
        if (d[i] >= reach) continue;
        for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
          if (j >= 0 && d[j] < 0) { d[j] = d[i] + 1; queue.push(j); }
        }
      }
      return (x, y) => {
        const sx = Math.min(w - 1, Math.max(0, Math.round(x / back))), sy = Math.min(h - 1, Math.max(0, Math.round(y / back)));
        const v = d[sy * w + sx];
        return v > 0;
      };
    })();
    const blobs = diamondBlobs(img, inCloth).filter(([x, y]) => band(x, y));
    const visible = (at) => at[0] >= 0 && at[1] >= 0 && at[0] < img.width && at[1] < img.height;
    const match = (H) => {
      const out = [], used = new Set();
      for (const d of DIAMONDS) {
        const at = apply(H, d);
        if (!visible(at)) continue;
        const along = Math.abs(d[1]) > W ? [L / 8, 0] : [0, W / 4];
        const gate = Math.max(10, 0.4 * dist(at, apply(H, [d[0] + (d[0] < L / 2 ? 1 : -1) * along[0], d[1] + (d[1] < W / 2 ? 1 : -1) * along[1]])));
        let best = -1, bd = gate;
        blobs.forEach((p, k) => { const e = dist(p, at); if (e < bd && !used.has(k)) { bd = e; best = k; } });
        if (best >= 0) { used.add(best); out.push([d, blobs[best], bd / gate]); }
      }
      return out;
    };
    const c0 = [img.width / 2, img.height / 2];
    const CORNERS = [[-CUSHION, -CUSHION], [L + CUSHION, -CUSHION], [L + CUSHION, W + CUSHION], [-CUSHION, W + CUSHION]];
    // 기준 모서리 V(사진), 장축 방향 변 xl, 단축 방향 변 yl, 평행한 맞은편 변 (xl 쪽이면 W 만큼, yl 쪽이면 L 만큼 떨어져 있다)
    const solve = (V, xl, yl, partner, partnerOfX, f) => {
      const Kinv = [1 / f, 0, -c0[0] / f, 0, 1 / f, -c0[1] / f, 0, 0, 1];
      const omega = mul([Kinv[0], 0, 0, 0, Kinv[4], 0, Kinv[2], Kinv[5], 1], Kinv);
      const Vh = h3(V);
      let vx, vy;
      if (partnerOfX) { vx = cross3(xl, partner); vy = cross3(yl, mv(omega, vx)); }
      else { vy = cross3(yl, partner); vx = cross3(xl, mv(omega, vy)); }
      const nk = (v) => norm3(mv(Kinv, v));
      let alpha, beta;
      if (partnerOfX) { beta = -dot(partner, Vh) / ((W + 2 * CUSHION) * dot(partner, vy)); alpha = beta * nk(vy) / (nk(vx) || 1); }
      else { alpha = -dot(partner, Vh) / ((L + 2 * CUSHION) * dot(partner, vx)); beta = alpha * nk(vx) / (nk(vy) || 1); }
      if (!Number.isFinite(alpha) || !Number.isFinite(beta)) return null;
      const out = [];
      // α의 부호는 직교 조건이 안 정한다 — 둘 다 놓아 보고, 네 모서리 자리마다 (모서리 기준 좌표 → 세계)
      for (const sa of [1, -1]) {
        const Hp = [sa * alpha * vx[0], beta * vy[0], Vh[0], sa * alpha * vx[1], beta * vy[1], Vh[1], sa * alpha * vx[2], beta * vy[2], Vh[2]];
        for (const [cx, cy] of CORNERS) {
          const sx = cx < 0 ? 1 : -1, sy = cy < 0 ? 1 : -1;          // 모서리에서 탁자 안쪽 방향
          const T = [sx, 0, -sx * cx, 0, sy, -sy * cy, 0, 0, 1];      // 세계 → 모서리 기준
          const H = mul(Hp, T);
          // 거울상 걸러내기: 위에서 본 그림이면 세계 +x, +y의 화면 방향이 cross < 0 (화면 y가 아래로)
          const o = apply(H, [L / 2, W / 2]), ex = apply(H, [L / 2 + 100, W / 2]), ey = apply(H, [L / 2, W / 2 + 100]);
          const cr = (ex[0] - o[0]) * (ey[1] - o[1]) - (ex[1] - o[1]) * (ey[0] - o[0]);
          // 변 방향도 맞아야 한다: 모서리에서 탁자 안쪽으로 가면 실제 변을 따라가야 한다
          const inward = apply(H, [cx + sx * 300, cy]);
          if (cr < 0 && Number.isFinite(inward[0])) out.push(H);
        }
      }
      return out;
    };
    let pick = null;
    const lineAway = (ln, p) => Math.abs(ln[0] * p[0] + ln[1] * p[1] + ln[2]);
    // ★찾은 변은 모두 탁자 테두리(쿠션 바깥 끝) 위에 있어야 한다 (2026-10-04). 다이아몬드 개수만 보니
    // 실제 사진에서 "어느 변이 어느 변과 평행한가"를 틀린 가설도 같은 수를 맞춰 이겼다. 변마다 양 끝이
    // 테두리 네 변 중 하나의 그림 위 직선에 가까운지 보고, 설명 안 되는 변 하나에 다이아몬드 3개만큼 깎는다.
    const segs = lines.map((x) => [[x.a[0] * back, x.a[1] * back], [x.b[0] * back, x.b[1] * back]]);
    const tol = 0.02 * Math.max(img.width, img.height);
    const unexplained = (H) => {
      const edges = [[CORNERS[0], CORNERS[1]], [CORNERS[1], CORNERS[2]], [CORNERS[2], CORNERS[3]], [CORNERS[3], CORNERS[0]]]
        .map(([p, q]) => { const a = apply(H, p), b = apply(H, q); return cross3(h3(a), h3(b)); })
        .map((l) => { const s = Math.hypot(l[0], l[1]) || 1; return [l[0] / s, l[1] / s, l[2] / s]; });
      let bad = 0;
      for (const [a, b] of segs) {
        if (!edges.some((l) => lineAway(l, a) < tol && lineAway(l, b) < tol)) bad++;
      }
      return bad;
    };
    for (const cn of corners) {
      const A = lines[cn.p], B = lines[cn.q];
      for (const [xl, yl] of [[A, B], [B, A]]) {
        for (const other of lines) {
          if (other === A || other === B) continue;
          // 맞은편 변: 기준 모서리를 지나지 않아야 한다
          if (lineAway(other.l, cn.at) < 0.05 * img.width) continue;
          for (const partnerOfX of [true, false]) {
            for (let f = 0.35 * img.width; f <= 2.2 * img.width; f *= 1.04) {
              const hs = solve(cn.at, xl.l, yl.l, other.l, partnerOfX, f);
              if (!hs) continue;
              for (const H of hs) {
                if (!H.every(Number.isFinite)) continue;
                const mt = match(H);
                const s = mt.length - 0.5 * mt.reduce((acc, x) => acc + x[2], 0) / Math.max(1, mt.length) - 3 * unexplained(H);
                if (opts.debug) opts.debug.push({ s, n: mt.length, corner: cn.at.map(Math.round), xl: lines.indexOf(xl), yl: lines.indexOf(yl), partner: lines.indexOf(other), partnerOfX, f: Math.round(f) });
                if (!pick || s > pick.score) pick = { H, score: s, corner: cn.at };
              }
            }
          }
        }
      }
    }
    if (!pick) return { error: "맞은편 레일이 안 보여 위치를 정하지 못했습니다. 모서리 하나와 맞은편 레일이 함께 나오게 찍어 주세요." };
    let H = pick.H, matched = [];
    for (let round = 0; round < 3; round++) {
      matched = match(H);
      if (matched.length < 4) break;
      const anchor = [[CORNERS.reduce((b, c) => (dist(apply(H, c), pick.corner) < dist(apply(H, b), pick.corner) ? c : b)), pick.corner, 4]];
      const next = fitHomography(anchor.concat(matched.map(([d, p]) => [d, p, 1])));
      if (!next) break;
      H = next;
    }
    if (matched.length < 4) warnings.push("레일의 다이아몬드 점을 몇 개 못 찾았습니다 — 위치가 덜 정확할 수 있습니다.");
    const top = warpWorld(img, H);
    const { found, warnings: w2 } = detectBalls(top);
    warnings.push(...w2.filter((s) => !/찾지 못했습니다/.test(s)));
    // 사진을 찍은 사람이 서 있던 쪽 = **사진 맨 아래 가운데가 가리키는 탁자 위 지점** (멀수록 펴진 그림이 성기다).
    // 처음엔 가설의 기준 모서리를 썼는데, 실제 사진에서 그것이 맞은편(먼) 모서리로 잡혀 먼 쪽 사진을 더 믿었다.
    const Hi = inv(H);
    const foot = Hi ? apply(Hi, [img.width / 2, img.height - 1]) : [0, 0];
    const anchorWorld = [Math.max(-CUSHION, Math.min(L + CUSHION, foot[0])), Math.max(-CUSHION, Math.min(W + CUSHION, foot[1]))];
    return { H, corners: [pick.corner], diamonds: matched.map(([, p]) => p), blobs, lines: lines.map((x) => x.l),
             balls: found, anchor: anchorWorld, warnings, top, score: pick.score };
  }
  // 예전 이름 (화면이 쓴다)
  const analyzeEnd = (img) => analyzeView(img);

  // 여러 장을 합친다. 첫 장의 틀을 기준으로, 다음 장은 그대로 또는 180도 — 공이 더 잘 겹치는 쪽.
  // 공은 자기 사진의 기준 모서리에 가까울수록 믿는다.
  function combineEnds(...views) {
    const names = { white: "흰공", yellow: "노란공", red: "빨간공" };
    const flip = (p) => [L - p[0], W - p[1]];
    const placed = [];
    views.forEach((v, k) => {
      let turn = false;
      if (k > 0) {
        const cost = (t) => {
          let s = 0, n = 0;
          for (const c of Object.keys(v.balls)) {
            const ref = placed[0].v.balls[c];
            if (!ref) continue;
            const a = t ? flip(v.balls[c].world) : v.balls[c].world;
            s += Math.min(dist(a, ref.world), 600); n++;
          }
          return n ? s / n : 0;
        };
        turn = cost(true) < cost(false);
      }
      placed.push({ v, turn });
    });
    const out = {}, warnings = [];
    for (const colour of ["white", "yellow", "red"]) {
      const cand = [];
      for (const { v, turn } of placed) {
        const b = v.balls && v.balls[colour];
        if (!b) continue;
        const at = turn ? flip(b.world) : b.world;
        const near = turn ? flip(v.anchor) : v.anchor;
        cand.push({ at, wt: b.confidence / (1 + (dist(at, near) / 1300) ** 2), conf: b.confidence });
      }
      if (!cand.length) { warnings.push(`${names[colour]}을 사진 어디에서도 찾지 못했습니다.`); continue; }
      const best = cand.reduce((a, c) => (c.wt > a.wt ? c : a));
      const close = cand.filter((c) => dist(c.at, best.at) < 200);
      if (close.length < cand.length) warnings.push(`${names[colour]}이 사진마다 다른 자리로 잡혔습니다 — 가까이 찍힌 쪽을 썼습니다.`);
      const s = close.reduce((a, c) => a + c.wt, 0);
      const at = [close.reduce((a, c) => a + c.at[0] * c.wt, 0) / s, close.reduce((a, c) => a + c.at[1] * c.wt, 0) / s];
      out[colour] = { at: [Math.max(R, Math.min(L - R, at[0])), Math.max(R, Math.min(W - R, W - at[1]))],
                      confidence: Math.max(...cand.map((c) => c.conf)) };
    }
    return { balls: out, warnings, turns: placed.map((p) => p.turn) };
  }

  function analyzePair(img1, img2) {
    const ends = [analyzeView(img1), analyzeView(img2)];
    const bad = ends.find((e) => e.error);
    if (bad) return { balls: {}, warnings: [bad.error], ends };
    const r = combineEnds(ends[0], ends[1]);
    return { balls: r.balls, warnings: [...ends.flatMap((e) => e.warnings), ...r.warnings], ends, turns: r.turns };
  }

  // 조언판 좌표(mm, y 아래로) → 사진 픽셀. 화면이 사진 위에 공 자리를 그릴 때.
  function toImage(corners, at) {
    const H = homography(WORLD, corners);
    return H ? apply(H, [at[0], W - at[1]]) : null;
  }

  // 화면이 쓰는 것: 펴진 그림 픽셀과 조언판 좌표의 관계 (배경으로 깔 때)
  const topFrame = { width: TOP_W, height: TOP_H, pad: PAD, px: PX };
  return { analyze, analyzeView, analyzeEnd, analyzePair, combineEnds, warpWorld, toImage, findTable, pickAssignment, warp, detectBalls, estimateCamera, correctParallax,
           homography, apply, inv, orderCyclic, shrink, topFrame, L, W, R };
})();

if (typeof module !== "undefined") module.exports = PHOTO;
