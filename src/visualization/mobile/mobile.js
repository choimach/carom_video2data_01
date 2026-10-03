// 폰 앱 전용 — 탁자 오른쪽 기둥에 공략 선택 단추 넷을 둔다 (2026-10-03).
//
// 선수: "phone app에서는 수구 버튼 말고 1적구, 면 자동 버튼은 없어도 되고, 다시 놓기와 실행
// 버튼만 있으면 돼. 그리고 공략법 선택 버튼 4개는 다시 살려야지!" — 가로 화면에서 공략 카드
// 넷이 겨냥 그림과 설명 아래로 한참 밀려나 안 보였다. 카드와 같은 넷(득점 확률 1등 + 프로
// 순위 1~3)을 탁자 옆에 짧게 둔다. 카드는 아래에 그대로 있다 (판단 단추가 거기 있다).
//
// 페이지의 render()를 감싼다 — 그릴 때마다 단추도 다시 만든다.
(() => {
  const bar = document.querySelector(".table-card .bar");
  const run = document.getElementById("run");
  if (!bar || !run || typeof render !== "function") return;
  const box = document.createElement("div");
  box.id = "picks";
  bar.insertBefore(box, document.getElementById("clear"));

  const short = (row) => label(row);   // 카드와 같은 이름 (공략 · 1적구 면)
  const paintPicks = () => {
    box.textContent = "";
    if (!result || !result.routes.length) return;
    const byChance = [...result.routes].sort((a, b) => b.chance - a.chance)[0];
    const slots = [
      [byChance, `득점 ${Math.round(byChance.chance * 100)}%`],
      ...result.routes.slice(0, 3).map((row, i) => [row, `${i + 1}순위`]),
    ];
    for (const [row, tag] of slots) {
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `<small>${tag}</small>${short(row)}`;
      b.setAttribute("aria-pressed", String(row.key === picked));
      b.onclick = () => { stopPlaying(); picked = row.key; render(); };
      box.appendChild(b);
    }
  };
  const original = render;
  render = function () { original.apply(this, arguments); paintPicks(); };
  paintPicks();
})();
