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
    return { small, w, h, big, edge, mask };
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
  const DIAMONDS = (() => {
    const out = [];
    for (let k = 1; k <= 7; k++) { out.push([k * L / 8, -CUSHION - DIAMOND]); out.push([k * L / 8, W + CUSHION + DIAMOND]); }
    for (let j = 1; j <= 3; j++) out.push([-CUSHION - DIAMOND, j * W / 4]);
    return out;
  })();
  // 밝고 채도 낮은 작은 점 (천 밖) — 다이아몬드 후보
  function diamondBlobs(img, clothMaskFull) {
    const { width: w, height: h, data } = img;
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (clothMaskFull(i % w, (i - (i % w)) / w)) continue;
      const p = hsv(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
      mask[i] = p[2] > 160 && p[1] < 110 ? 1 : 0;
    }
    const out = [];
    for (const c of components(mask, w, h)) {
      const n = c.pixels.length;
      if (n < 3 || n > 900) continue;
      let sx = 0, sy = 0;
      for (const p of c.pixels) { sx += p % w; sy += (p - (p % w)) / w; }
      out.push([sx / n + 0.5, sy / n + 0.5]);
    }
    return out;
  }

  // 사진 한 장(탁자 한쪽 끝) → 자기 틀의 공 자리. corners를 주면 [A, D] 두 모서리로 쓴다.
  function analyzeEnd(img, given = null, opts = {}) {
    const warnings = [];
    const region = clothRegion(img);
    if (region.error) return { error: region.error };
    const back = 1 / region.small.scale;
    const { w, h, edge } = region;
    const H0 = hull(edge);
    const m = 2.5;
    const onBorder = (p) => p[0] <= m || p[1] <= m || p[0] >= w - m || p[1] >= h - m;
    // 껍질에서 화면 가장자리를 따라가는 변을 끊어 열린 사슬로
    let cut = -1;
    for (let i = 0; i < H0.length; i++) { if (onBorder(H0[i]) && onBorder(H0[(i + 1) % H0.length])) { cut = i; break; } }
    if (cut < 0) return { whole: true };                    // 가장자리에 안 걸림 = 탁자 전체가 보인다
    let chain = H0.slice(cut + 1).concat(H0.slice(0, cut + 1));
    while (chain.length && onBorder(chain[0]) && chain.length > 1 && onBorder(chain[1])) chain.shift();
    while (chain.length > 1 && onBorder(chain[chain.length - 1]) && onBorder(chain[chain.length - 2])) chain.pop();
    if (chain.length < 4) return { error: "탁자의 끝(단쿠션과 두 모서리)이 보이지 않습니다. 한쪽 끝이 다 나오게 찍어 주세요." };
    // 각 변의 직선을 테두리 점들로 다시 맞춘다
    const segDist = (p, a, b) => { const l = dist(a, b); return l < 1e-9 ? dist(p, a) : Math.abs(cross(a, b, p)) / l; };
    const within = (p, a, b) => { const t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (dist(a, b) ** 2 || 1); return t > 0.08 && t < 0.92; };
    const lineOf = (a, b) => {
      const near = edge.filter((p) => !onBorder(p) && segDist(p, a, b) < 3 && within(p, a, b));
      const l = near.length >= 6 ? fitLine(near) : fitLine([a, b]);
      return [l[0], l[1], l[2] * back];                        // 원래 크기 픽셀로 (a, b는 그대로)
    };
    const meet = (l1, l2) => { const v = cross3(l1, l2); return [v[0] / v[2], v[1] / v[2]]; };
    const crossDir = (u, v) => u[0] * v[1] - u[1] * v[0];
    const big = (p) => [p[0] * back, p[1] * back];
    // 다이아몬드 후보 (천 밖의 밝고 채도 낮은 작은 점)
    const clothSmall = region.mask;
    const inCloth = (x, y) => {
      const sx = Math.min(w - 1, Math.max(0, Math.round(x / back))), sy = Math.min(h - 1, Math.max(0, Math.round(y / back)));
      return clothSmall[sy * w + sx] === 1;
    };
    const blobs = diamondBlobs(img, inCloth);
    // 예측한 다이아몬드와 찾은 점 짝짓기 (하나에 하나). → [[다이아몬드, 점, 거리/문턱]…]
    const match = (H) => {
      const out = [], used = new Set();
      for (const d of DIAMONDS) {
        const at = apply(H, d);
        if (!(at[0] >= 0 && at[1] >= 0 && at[0] < img.width && at[1] < img.height)) continue;
        // 문턱 = 이웃 다이아몬드까지 화면 거리의 40%
        const along = Math.abs(d[1]) > W ? [L / 8, 0] : [0, W / 4];
        const gate = Math.max(10, 0.4 * dist(at, apply(H, [d[0] + along[0], d[1] + along[1]])));
        let best = -1, bd = gate;
        blobs.forEach((p, k) => { const e = dist(p, at); if (e < bd && !used.has(k)) { bd = e; best = k; } });
        if (best >= 0) { used.add(best); out.push([d, blobs[best], bd / gate]); }
      }
      return out;
    };
    const c0 = [img.width / 2, img.height / 2];
    const span = W + 2 * CUSHION;
    // 가설 하나: 사슬의 연속한 네 점 (앞 장쿠션 끝, 모서리, 모서리, 뒤 장쿠션 끝) → 가장 잘 맞는 H와 점수
    const tryHypothesis = (Q0, Q1, Q2, Q3) => {
      const lines = [lineOf(Q0, Q1), lineOf(Q1, Q2), lineOf(Q2, Q3)];
      let c1 = meet(lines[0], lines[1]), c2 = meet(lines[1], lines[2]);
      if (given) [c1, c2] = given.slice(0, 2).map((p) => p.slice());
      if (![...c1, ...c2].every(Number.isFinite)) return null;
      const s0 = big(Q0), s3 = big(Q3);
      // 어느 모서리가 A(−C,−C)인가: 위에서 본 거울상 아닌 그림이면 cross(x방향, y방향) < 0 (화면 y가 아래로)
      let A, D, longs, dirX;
      const dx1 = [s0[0] - c1[0], s0[1] - c1[1]], dy1 = [c2[0] - c1[0], c2[1] - c1[1]];
      if (crossDir(dx1, dy1) < 0) { A = c1; D = c2; longs = [lines[0], lines[2]]; dirX = dx1; }
      else { A = c2; D = c1; longs = [lines[2], lines[0]]; dirX = [s3[0] - c2[0], s3[1] - c2[1]]; }
      // 처음 추정 — 장쿠션 소실점 vx와, 화각 f에서 직교 조건으로 정해지는 단축 소실점 vy (AD 위).
      // ★화각을 가정하지 않는다 (2026-10-03): 사전값 0.72W로 풀었더니 합성 사진(실제 0.64W)에서 장축
      // 축척이 30% 틀려 다이아몬드를 엉뚱하게 짝지었다. f를 넓게 훑어 **다이아몬드 예측 자리와 찾은
      // 밝은 점이 가장 많이 겹치는** f를 고른다.
      const vx = cross3(longs[0], longs[1]);
      const Ah = h3(A), Dh = h3(D), lAD = cross3(Ah, Dh);
      const initialFor = (f) => {
        const Kinv = [1 / f, 0, -c0[0] / f, 0, 1 / f, -c0[1] / f, 0, 0, 1];
        const omega = mul([Kinv[0], 0, 0, 0, Kinv[4], 0, Kinv[2], Kinv[5], 1], Kinv);   // K⁻ᵀK⁻¹
        const vy = cross3(lAD, mv(omega, vx));
        const vyD = cross3(vy, Dh), AD = cross3(Ah, Dh);
        const beta = -dot(vyD, AD) / (dot(vyD, vyD) || 1) / span;      // span·β·vy + A ∝ D
        const nk = (v) => norm3(mv(Kinv, v));
        const alpha = beta * nk(vy) / (nk(vx) || 1);
        const build = (al) => mul([al * vx[0], beta * vy[0], Ah[0], al * vx[1], beta * vy[1], Ah[1], al * vx[2], beta * vy[2], Ah[2]],
                                  [1, 0, CUSHION, 0, 1, CUSHION, 0, 0, 1]);   // 세계 → A 기준
        let H = build(alpha);
        const probe = apply(H, [-CUSHION + 400, -CUSHION]);
        if ((probe[0] - A[0]) * dirX[0] + (probe[1] - A[1]) * dirX[1] < 0) H = build(-alpha);
        return H;
      };
      let H = null, score = -Infinity;
      for (let f = 0.35 * img.width; f <= 2.0 * img.width; f *= 1.03) {
        const cand = initialFor(f);
        if (!cand || !cand.every(Number.isFinite)) continue;
        const m = match(cand);
        const s = m.length - 0.5 * m.reduce((acc, x) => acc + x[2], 0) / Math.max(1, m.length);
        if (s > score) { score = s; H = cand; }
      }
      return H && { H, A, D, score };
    };
    // 사슬을 꼭짓점 4~6개로 줄여 연속한 네 점마다 가설을 세운다. 모서리가 셋 보이는 사진(모서리에서
    // 찍은 것)에서는 어느 둘이 같은 단쿠션인지 모른다 — 다이아몬드가 가장 잘 맞는 쪽을 고른다.
    let pick = null;
    for (const k of [4, 5, 6]) {
      if (chain.length < k) break;
      const V = reduceChain(chain, k);
      for (let s = 0; s + 3 < V.length; s++) {
        const hyp = tryHypothesis(V[s], V[s + 1], V[s + 2], V[s + 3]);
        if (hyp && (!pick || hyp.score > pick.score)) pick = hyp;
      }
    }
    if (!pick) return { error: "탁자의 끝(단쿠션과 두 모서리)을 읽지 못했습니다. 한쪽 끝이 다 나오게 찍어 주세요." };
    let { H, A, D } = pick;
    let matched = [];
    const initial = H;
    for (let round = 0; round < (opts.rounds ?? 3); round++) {
      matched = match(H);
      if (matched.length < 3) break;
      const pairs = [[[-CUSHION, -CUSHION], A, 6], [[-CUSHION, W + CUSHION], D, 6], ...matched.map(([d, p]) => [d, p, 1])];
      const next = fitHomography(pairs);
      if (!next) break;
      H = next;
    }
    if (matched.length < 3) warnings.push("레일의 다이아몬드 점을 거의 못 찾았습니다 — 먼 쪽 공은 덜 정확합니다.");
    const top = warpWorld(img, H);
    const { found, warnings: w2 } = detectBalls(top);
    warnings.push(...w2.filter((s) => !/찾지 못했습니다/.test(s)));
    return { H, initial, blobs, corners: [A, D], diamonds: matched.map(([, p]) => p), balls: found, warnings, top };
  }

  // 두 끝을 합친다. 두 번째 장은 반대쪽 끝이므로 180도 돌린다. → analyze()와 같은 모양의 balls
  function combineEnds(one, two) {
    const out = {}, warnings = [];
    const names = { white: "흰공", yellow: "노란공", red: "빨간공" };
    for (const colour of ["white", "yellow", "red"]) {
      const cand = [];
      // 자기 끝에서 가까울수록 믿는다 (멀면 펴진 그림이 성기고 원근 오차가 커진다)
      for (const [end, flip] of [[one, false], [two, true]]) {
        const b = end && end.balls && end.balls[colour];
        if (!b) continue;
        const [x, y] = b.world;
        const wt = b.confidence / (1 + ((x + CUSHION) / 900) ** 2);
        cand.push({ at: flip ? [L - x, W - y] : [x, y], wt, conf: b.confidence });
      }
      if (!cand.length) { warnings.push(`${names[colour]}을 두 사진 어디에서도 찾지 못했습니다.`); continue; }
      let at;
      if (cand.length === 2 && dist(cand[0].at, cand[1].at) < 200) {
        const s = cand[0].wt + cand[1].wt;
        at = [(cand[0].at[0] * cand[0].wt + cand[1].at[0] * cand[1].wt) / s, (cand[0].at[1] * cand[0].wt + cand[1].at[1] * cand[1].wt) / s];
      } else {
        if (cand.length === 2) warnings.push(`${names[colour]}이 두 사진에서 다른 자리로 잡혔습니다 — 확인해 주세요.`);
        at = cand.reduce((a, c) => (c.wt > a.wt ? c : a)).at;
      }
      out[colour] = { at: [Math.max(R, Math.min(L - R, at[0])), Math.max(R, Math.min(W - R, W - at[1]))],
                      confidence: Math.max(...cand.map((c) => c.conf)) };
    }
    return { balls: out, warnings };
  }

  // 두 장을 한 번에: 한 장에 탁자 전체가 보이면 그 장만으로 (한 장 방식), 아니면 두 끝을 합친다.
  function analyzePair(img1, img2) {
    const ends = [analyzeEnd(img1), analyzeEnd(img2)];
    const whole = [img1, img2].find((im, k) => ends[k].whole);
    if (whole) { const r = analyze(whole); return { balls: r.balls, warnings: r.warnings, ends, single: true }; }
    const bad = ends.find((e) => e.error);
    if (bad) return { balls: {}, warnings: [bad.error], ends };
    const r = combineEnds(ends[0], ends[1]);
    return { balls: r.balls, warnings: [...ends.flatMap((e) => e.warnings), ...r.warnings], ends };
  }

  // 조언판 좌표(mm, y 아래로) → 사진 픽셀. 화면이 사진 위에 공 자리를 그릴 때.
  function toImage(corners, at) {
    const H = homography(WORLD, corners);
    return H ? apply(H, [at[0], W - at[1]]) : null;
  }

  // 화면이 쓰는 것: 펴진 그림 픽셀과 조언판 좌표의 관계 (배경으로 깔 때)
  const topFrame = { width: TOP_W, height: TOP_H, pad: PAD, px: PX };
  return { analyze, analyzeEnd, analyzePair, combineEnds, warpWorld, toImage, findTable, pickAssignment, warp, detectBalls, estimateCamera, correctParallax,
           homography, apply, inv, orderCyclic, shrink, topFrame, L, W, R };
})();

if (typeof module !== "undefined") module.exports = PHOTO;
