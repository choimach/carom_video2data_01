#!/usr/bin/env bash
# 조언판을 헤드리스 크롬으로 띄워 무작위 배치 하나를 실행하고 탁자를 찍는다.
#
# sudo 없이 된다 (2026-09-30). 크롬에 빠진 libnspr4·libnss3를 `apt-get download`로 받아
# ~/.cache/carom_libs에 풀고 LD_LIBRARY_PATH로 준다. `!`로는 sudo 비밀번호를 못 넣는다.
#
#   bash tools/screenshot_assistant.sh [out.png]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-/tmp/assistant.png}
LIBS=~/.cache/carom_libs
SHELL_BIN=$(ls -d ~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell | tail -1)
CORE=$(find ~/.npm/_npx -maxdepth 3 -type d -name playwright-core | head -1)
if [ ! -f "$LIBS/usr/lib/x86_64-linux-gnu/libnss3.so" ]; then
  mkdir -p "$LIBS/debs" && (cd "$LIBS/debs" && apt-get download libnspr4 libnss3 >/dev/null)
  for f in "$LIBS"/debs/*.deb; do dpkg -x "$f" "$LIBS"; done
fi
cd "$REPO/build"
~/.venvs/carom/bin/python -m http.server 8765 >/dev/null 2>&1 & SERVER=$!
trap 'kill $SERVER' EXIT
sleep 1
node -e "
const { chromium } = require('$CORE');
(async () => {
  const b = await chromium.launch({ executablePath: '$SHELL_BIN',
    env: { ...process.env, LD_LIBRARY_PATH: '$LIBS/usr/lib/x86_64-linux-gnu' } });
  const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  p.on('console', m => m.type() === 'error' && errs.push(m.text()));
  await p.goto('http://localhost:8765/carom_assistant.html');
  await p.click('#dice'); await p.click('#run');
  await p.waitForTimeout(20000);
  await p.locator('.table-card').screenshot({ path: '$OUT' });
  console.log('$OUT', 'errors', JSON.stringify(errs));
  await b.close();
})();"
