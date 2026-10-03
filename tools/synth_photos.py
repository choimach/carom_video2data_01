"""사진 인식(photo.js)을 재는 합성 사진 — 정답 좌표를 아는 3D 장면을 폰 카메라로 찍는다.

실제 당구장 사진에는 정답이 없다. 그래서 장면을 만든다: 천(청색) · 쿠션(천 색, 높이 37 mm,
폭 50 mm) · 나무 레일 · 다이아몬드 · 공 셋(반지름 30.75 mm, 지면에서 떠 있는 구). 카메라는
사람이 서서 폰을 드는 자세 몇 가지 (장축 옆, 비스듬히, 단축 끝, 모서리, 높게, 낮게).

    ~/.venvs/carom/bin/python tools/synth_photos.py OUT_DIR [판 수]

OUT_DIR/NNN.png 와 OUT_DIR/truth.json (조언판 좌표: mm, y 아래로) 을 쓴다.
⚠️ 장면은 단순하다 (그림자·반사·조명 얼룩 없음). 여기서 잘 되는 것은 **필요조건**이다.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

L, W, R = 2844.0, 1422.0, 30.75
CUSHION, CUSHION_H = 50.0, 37.0
RAIL, RAIL_H = 140.0, 45.0
CLOTH = np.array([170, 95, 40], float)        # BGR — 청색 천
WOOD = np.array([40, 70, 115], float)
FLOOR = np.array([60, 60, 60], float)
BALLS = {"white": np.array([235, 240, 240], float), "yellow": np.array([40, 195, 235], float),
         "red": np.array([35, 30, 200], float)}
LIGHT = np.array([0.3, -0.2, 1.0]); LIGHT /= np.linalg.norm(LIGHT)

POSES = {   # 카메라 위치 (월드 mm, y 위로), 바라보는 점
    "long_mid": ((L / 2, -1150, 1650), (L / 2, W / 2, 0)),
    "long_off": ((900, -1000, 1600), (L / 2 + 150, W / 2, 0)),
    "short_end": ((-1250, W / 2, 1800), (L / 2 - 100, W / 2, 0)),
    "corner": ((-650, -700, 1750), (L / 2, W / 2, 0)),
    "high": ((L / 2, -500, 2300), (L / 2, W / 2, 0)),
    "low": ((L / 2, -1500, 1300), (L / 2, W / 2, 0)),
}


def camera(pos, look, w, h, fov_w=1.32):        # 가로 화각 약 76도
    c = np.array(pos, float); f = w / 2 / math.tan(fov_w / 2)
    z = np.array(look, float) - c; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)                             # 이미지 아래쪽
    return c, f, x, y, z


def project(p, cam, w, h):
    c, f, x, y, z = cam
    d = np.asarray(p, float) - c
    return np.array([w / 2 + f * d @ x / (d @ z), h / 2 + f * d @ y / (d @ z)])


def render(balls, cam, w, h, rng):
    c, f, x, y, z = cam
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = (z[None, None] + ((u - w / 2) / f)[..., None] * x + ((v - h / 2) / f)[..., None] * y)
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    img = np.empty((h, w, 3)); img[:] = FLOOR
    depth = np.full((h, w), np.inf)

    def plane(zh, region, colour, shade=1.0):
        t = (zh - c[2]) / rays[..., 2]
        p = c + rays * t[..., None]
        ok = (t > 0) & region(p[..., 0], p[..., 1]) & (t < depth)
        img[ok] = colour * shade; depth[ok] = t[ok]
        return p, ok

    def ring(px, py, inner, outer):
        out = (px > -outer) & (px < L + outer) & (py > -outer) & (py < W + outer)
        inn = (px > -inner) & (px < L + inner) & (py > -inner) & (py < W + inner)
        return out & ~inn

    p, ok = plane(RAIL_H, lambda px, py: ring(px, py, CUSHION, CUSHION + RAIL), WOOD)
    # 다이아몬드: 장축 8칸, 단축 4칸, 레일 가운데
    mid = CUSHION + RAIL / 2
    dots = [(i * L / 8, -mid) for i in range(1, 8)] + [(i * L / 8, W + mid) for i in range(1, 8)] \
        + [(-mid, j * W / 4) for j in range(1, 4)] + [(L + mid, j * W / 4) for j in range(1, 4)]
    for dx, dy in dots:
        hit = ok & (np.hypot(p[..., 0] - dx, p[..., 1] - dy) < 9)
        img[hit] = (200, 225, 235)
    plane(CUSHION_H, lambda px, py: ring(px, py, 0, CUSHION), CLOTH, 0.82)
    plane(0.0, lambda px, py: (px >= 0) & (px <= L) & (py >= 0) & (py <= W), CLOTH)
    for name, (bx, by) in balls.items():
        o = c - np.array([bx, by, R])
        b = rays @ o; cc = o @ o - R * R
        disc = b * b - cc
        t = -b - np.sqrt(np.maximum(disc, 0))
        hit = (disc > 0) & (t > 0) & (t < depth)
        n = (c + rays * t[..., None]) - np.array([bx, by, R]); n /= np.linalg.norm(n, axis=2, keepdims=True)
        lam = np.clip(n @ LIGHT, 0, 1)
        spec = np.clip(((2 * (n @ LIGHT))[..., None] * n - LIGHT) @ -rays.reshape(-1, 3).T[:, 0] if False else 0, 0, 1)
        col = BALLS[name][None, None] * (0.35 + 0.65 * lam)[..., None]
        img[hit] = col[hit]; depth[hit] = t[hit]
    img += rng.normal(0, 4, img.shape)               # 센서 잡음
    return np.clip(img, 0, 255).astype(np.uint8)


def main(out, n_per_pose=5):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    w, h = 1600, 1200
    truth = {}
    k = 0
    for pose, (pos, look) in POSES.items():
        cam = camera(pos, look, w, h)
        corners = [project((px, py, RAIL_H), cam, w, h) for px, py in
                   [(-CUSHION - RAIL, -CUSHION - RAIL), (L + CUSHION + RAIL, -CUSHION - RAIL),
                    (L + CUSHION + RAIL, W + CUSHION + RAIL), (-CUSHION - RAIL, W + CUSHION + RAIL)]]
        inside = all(0 <= cx < w and 0 <= cy < h for cx, cy in corners)
        for _ in range(n_per_pose):
            while True:
                pts = rng.uniform([R, R], [L - R, W - R], (3, 2))
                if min(np.linalg.norm(pts[i] - pts[j]) for i in range(3) for j in range(i + 1, 3)) > 3 * R:
                    break
            balls = dict(zip(["white", "yellow", "red"], map(tuple, pts)))
            cv2.imwrite(os.path.join(out, f"{k:03d}.png"), render(balls, cam, w, h, rng))
            # 조언판 좌표는 y가 아래로: (x, W − y)
            truth[f"{k:03d}"] = {"pose": pose, "whole_table": inside, "camera_mm": pos,
                                 "balls": {c: [round(x, 1), round(W - y, 1)] for c, (x, y) in balls.items()}}
            k += 1
    json.dump(truth, open(os.path.join(out, "truth.json"), "w"), indent=1)
    print(f"{k}장 → {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
