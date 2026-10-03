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
  // ★고른 공략의 두께 · 당점 · 강도 그림도 탁자 옆에 (선수, 2026-10-03: "공략시 당점, 두께,
  // 스트로크 강도 이미지도 table의 우측에"). 아래의 큰 그림(#aim)을 **베껴** 작게 보인다 —
  // 그림을 두 벌로 그리지 않는다. 끌어서 바꾸는 것은 아래 큰 그림에서 (여기는 보기만).
  const mini = document.createElement("div");
  mini.id = "mini";
  bar.insertBefore(mini, box);
  const paintMini = () => {
    mini.textContent = "";
    const row = result && chosenRow();
    const aim = document.querySelector("#aim .plate svg:not(.strength)");
    const power = document.querySelector("#aim svg.strength");
    if (!row || !aim) return;
    const add = (node, cls) => { const n = node.cloneNode(true); n.classList.add(cls); mini.appendChild(n); };
    add(aim, "mini-aim");
    const say = document.createElement("div");
    say.className = "mini-say";
    say.innerHTML = `두께 <b>${asThickness(row.hit.thickness)}</b> · 당점 <b>${asTip(row.hit.side, row.hit.vertical || 0)}</b>`;
    mini.appendChild(say);
    if (power) add(power, "mini-power");
    const said = document.createElement("div");
    said.className = "mini-say";
    said.innerHTML = `강도 <b>${SIM.strengthOf(row.hit.speed).toFixed(1)}</b>`;
    mini.appendChild(said);
  };

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
      b.innerHTML = `<b>${tag}</b> ${short(row)}`;
      b.title = `${tag} ${short(row)}`;   // 한 줄에 안 들어가면 잘린다 — 전체 이름은 아래 카드에
      b.setAttribute("aria-pressed", String(row.key === picked));
      b.onclick = () => { stopPlaying(); picked = row.key; render(); };
      box.appendChild(b);
    }
  };
  const original = render;
  render = function () { original.apply(this, arguments); paintMini(); paintPicks(); };
  paintMini(); paintPicks();
})();
