"""폰 앱(설치형 웹앱)을 조언판에서 만든다 → build/mobile/.

조언판(`build/carom_assistant.html` + sim·route·choice.js)을 그대로 쓰고, 그 위에
폰에서만 필요한 것을 붙인다: 매니페스트, 아이콘, 서비스 워커, Firestore로 기록을
보내는 `cloud.js`. 조언판(claude.ai)에 올리는 판에는 이것들이 없다.

    ~/.venvs/carom/bin/python tools/build_assistant.py
    ~/.venvs/carom/bin/python tools/build_mobile.py
    firebase deploy --only hosting:board --project carom001      # firebase.json 참조

⚠️ 같은 Firebase 프로젝트(carom001)에 2026-09-10의 폰 앱이 https://carom001.web.app 에
   있다 (저장소 carom_bot_01). 이쪽은 **다른 사이트**로 올린다 — 그 주소에 덮어쓰지 말 것.
"""

import datetime
import os
import shutil
import subprocess

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
MOBILE = os.path.join(ROOT, "src", "visualization", "mobile")
OUT = os.path.join(BUILD, "mobile")


def icon(size):
    """천 위의 공 셋 — 따로 그림 파일을 두지 않고 그린다."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (82, 107, 47)                       # BGR — 천 색 #2f6b52
    r = int(size * 0.13)
    for (x, y), colour in (((0.34, 0.40), (245, 245, 245)), ((0.66, 0.36), (40, 170, 230)),
                           ((0.50, 0.66), (50, 60, 190))):
        cv2.circle(img, (int(x * size), int(y * size)), r, colour, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x * size), int(y * size)), r, (30, 40, 30), max(1, size // 128), cv2.LINE_AA)
    return img


def main():
    page = os.path.join(BUILD, "carom_assistant.html")
    if not os.path.exists(page):
        raise SystemExit("먼저 tools/build_assistant.py를 돌리세요")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "icons"))
    for name in ("sim.js", "route.js", "choice.js", "search.js"):
        shutil.copy(os.path.join(BUILD, name), OUT)
    for name in ("cloud.js", "sw.js", "manifest.webmanifest"):
        shutil.copy(os.path.join(MOBILE, name), OUT)
    for size in (192, 512):
        cv2.imwrite(os.path.join(OUT, "icons", f"icon-{size}.png"), icon(size))

    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M} {commit}".strip()
    html = open(page, encoding="utf-8").read()
    head = ('<link rel="manifest" href="./manifest.webmanifest">\n'
            '<meta name="theme-color" content="#2f6b52">\n'
            '<link rel="icon" href="./icons/icon-192.png">\n'
            '<link rel="apple-touch-icon" href="./icons/icon-192.png">\n')
    marker = '<title>'
    assert marker in html
    html = html.replace(marker, head + marker, 1)
    html += (f'\n<script>window.CAROM_BUILD = {stamp!r};</script>\n'
             '<script type="module" src="./cloud.js"></script>\n')
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(html)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(OUT) for f in fs)
    print(f"{OUT} — {stamp} · {size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
