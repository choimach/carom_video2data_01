// 이 공략이 실제로 들어갈 확률 — 프로가 골랐는지와 무관하게.
//
// 왜 이것이 필요한가. "어느 프로가 무엇을 골랐나"는 프로끼리도 39%만 일치하는
// 과녁이고, 우리 모델은 이미 그 한가운데(40%)에 있다. 판을 열 배 모아도
// 프로들이 서로 더 일치하게 되지는 않는다. 반면 **들어갔는가는 다툼의 여지가
// 없다** — 그리고 선수가 정한 프로의 1순위 기준이 바로 득점 확률이다.
//
// 재는 법: 사람이 실제로 내는 만큼의 오차를 겨냥·강도·당점에 얹어 여러 번
// 치고, 몇 번 들어가는지 센다. **프로 데이터가 필요 없다** — 모든 후보에
// 대해 계산된다. 지금 순위에서 가장 무거운 `줄두께`는 이것의 조잡한 대용품이다.
//
// 오차의 크기는 손으로 정하지 않는다. 프로 2,150판의 실제 득점률(61%)에
// 맞도록 **맞춘다** (tools/fit_probability.js).
//
//   node tools/fit_probability.js          # 오차 크기를 맞추고 채점한다

const fs = require('fs');
const path = require('path');

const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

// 같은 씨앗이면 같은 답이 나와야 한다 — 두 설정을 견줄 때 잡음이 달라지면
// 무엇 때문에 달라졌는지 알 수 없다.
function noise(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 4294967296;
  };
}

// 두 개의 균등난수에서 정규난수 하나 (Box-Muller).
function gauss(next) {
  const u = Math.max(next(), 1e-12), v = next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function kissed(shot) {
  const balls = shot.events.filter((e) => e.kind === 'ball');
  if (!balls.length) return false;
  const first = balls[0];
  const second = balls.find((e) => e.detail !== first.detail);
  if (!second) return false;
  if (shot.events.some((e) => e.kind === 'kiss' && e.at <= second.at)) return true;
  return balls.some((e) => e !== first && e.detail === first.detail && e.at < second.at);
}

/**
 * 들어갈 확률.
 *
 * @param line  {deg, speed, side, up} — 겨냥 각도(도), 초속(mm/s), 당점(팁)
 * @param wobble {aim, speed, tip} — 오차의 표준편차 (도 · 비율 · 팁)
 * @param tries  몇 번 쳐 볼 것인가
 */
function chance(layout, cue, line, wobble, tries = 60, seed = 1) {
  const next = noise(seed);
  let made = 0;
  for (let i = 0; i < tries; i++) {
    const deg = line.deg + gauss(next) * wobble.aim;
    const speed = line.speed * (1 + gauss(next) * wobble.speed);
    const side = (line.side || 0) + gauss(next) * wobble.tip;
    const up = (line.up || 0) + gauss(next) * wobble.tip;
    // 미스큐 한계 밖은 칠 수 없다. 방향은 두고 길이만 줄인다.
    const out = Math.hypot(side, up);
    const keep = out > SIM.MAX_TIPS ? SIM.MAX_TIPS / out : 1;
    const rad = deg * Math.PI / 180;
    const shot = SIM.play(layout, cue, [Math.cos(rad) * speed, Math.sin(rad) * speed],
                          side * keep, up * keep);
    shot.cue = cue;
    const judged = SIM.judge(shot);
    if (judged.scored && !kissed(shot)) made++;
  }
  return made / tries;
}

module.exports = { chance, SIM };
