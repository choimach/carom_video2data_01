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
import json
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
MOBILE = os.path.join(ROOT, "src", "visualization", "mobile")
OUT = os.path.join(BUILD, "mobile")
# 올리는 판을 줄이는 도구 — 저장소 밖에 둔다: npm install --prefix ~/.cache/carom_tools esbuild@0.24.0
ESBUILD = os.environ.get("CAROM_ESBUILD", os.path.expanduser("~/.cache/carom_tools/node_modules/.bin/esbuild"))


# ★폰 앱은 누구나 받아 볼 수 있다 (선수 2026-10-24: 분석하면 AI로 만든 것·SOOP에서 받은 것이 보이나,
# 복사하기 쉬운가). 원본(src/)과 조언판(claude.ai)은 그대로 두고 **올리는 판만** 바꾼다:
#   · 주석을 지우고 줄인다 — 주석에 대화 인용·날짜·claude.ai가 1,000줄 가까이 있었다.
#   · 경기 이름을 뜻 없는 번호로 — "201755361_POWC2026"의 숫자가 SOOP VOD 번호였다.
#     조언판은 경기 이름을 "같은 경기인가"(excludeMatch)로만 쓰므로 번호로 바꿔도 같다.
# 복사 자체는 막을 수 없다 — 서버 없이 폰에서 도는 웹앱이라 받은 것이 곧 전부다.
def squeeze(code, loader):
    """주석을 지우고 줄인다. 이름 없는 형식(--format 없음)이라 맨 위 이름은 그대로 둔다 —
    sim.js · route.js · search.js가 서로의 맨 위 이름을 부르고, 워커도 그것을 읽는다."""
    done = subprocess.run([ESBUILD, f"--loader={loader}", "--minify", "--legal-comments=none",
                           "--charset=utf8", "--log-level=warning"],
                          input=code, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"esbuild 실패 ({loader}):\n{done.stderr[-2000:]}")
    return done.stdout.strip()


def blank_matches(html):
    """자료 안의 경기 이름을 g1, g2, … 로. 같은 경기는 같은 번호."""
    tag = '<script id="data" type="application/json">'
    start = html.index(tag) + len(tag)
    end = html.index("</script>", start)
    data = json.loads(html[start:end])
    names = {}
    for play in data["plays"]:
        if play.get("match") is not None:
            play["match"] = names.setdefault(play["match"], f"g{len(names) + 1}")
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return html[:start] + body + html[end:], len(names)


def squeeze_page(html):
    """페이지 안의 <style>·<script>(src 없는 것)를 줄이고 HTML 주석을 지운다."""
    def one(m):
        open_tag, body, close = m.group(1), m.group(3), m.group(4)
        if m.group(2) == "style":
            return open_tag + squeeze(body, "css") + close
        if "src=" in open_tag or "application/json" in open_tag or not body.strip():
            return m.group(0)
        return open_tag + squeeze(body, "js") + close
    parts = re.split(r"(<(?:script|style)[^>]*>.*?</(?:script|style)>)", html, flags=re.S)
    for i in range(0, len(parts), 2):                    # 태그 바깥 HTML만
        parts[i] = re.sub(r"<!--.*?-->", "", parts[i], flags=re.S)
    html = "".join(parts)
    return re.sub(r"(<(script|style)[^>]*>)(.*?)(</\2>)", one, html, flags=re.S)


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
    for name in ("sim.js", "route.js", "choice.js", "search.js", "photo.js"):
        shutil.copy(os.path.join(BUILD, name), OUT)
    for name in ("cloud.js", "sw.js", "manifest.webmanifest", "mobile.css", "mobile.js"):
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
    # 폰 화면 규칙은 페이지 스타일 **뒤에** 와야 이긴다 — 첫 스크립트 앞에 넣는다.
    first_script = '<script src="sim.js"></script>'
    assert first_script in html
    html = html.replace(first_script, '<link rel="stylesheet" href="./mobile.css">\n' + first_script, 1)
    html += (f'\n<script>window.CAROM_BUILD = {stamp!r};</script>\n'
             '<script src="./mobile.js"></script>\n'
             '<script type="module" src="./cloud.js"></script>\n')
    html, games = blank_matches(html)
    html = squeeze_page(html)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(html)
    for name in os.listdir(OUT):
        loader = {".js": "js", ".css": "css"}.get(os.path.splitext(name)[1])
        if loader:
            path = os.path.join(OUT, name)
            code = squeeze(open(path, encoding="utf-8").read(), loader)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(code)
    print(f"경기 이름 {games}개를 번호로, 주석을 지우고 줄임")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _s, fs in os.walk(OUT) for f in fs)
    print(f"{OUT} — {stamp} · {size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
