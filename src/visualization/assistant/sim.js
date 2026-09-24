// The table, ported from src/physics/spin.py and src/physics/simulator.py so a
// phone can play a shot without asking anything. Units are millimetres and
// seconds, with the nose line from (0,0) to (2844,1422).
//
// A ball carries two things besides where it is and how fast it is going:
//
//   slip - the velocity of the point touching the cloth. Friction acts on it,
//          and a struck ball slides on it before it rolls.
//   side - spin about the vertical axis, which is what 당점 puts on, what a
//          cushion reads off the ball, and what throws an object ball off the
//          line joining the centres.
//
// 당점 is given in tips: a cue tip is 12 mm against a 61.5 mm ball, so three
// tips is about half the radius, which is where a cue starts to miscue.
//
// The constants are the cloth's own except where this project measured or
// fitted its own: a cue ball opening at 2699 mm/s - the median over 1596
// tracked plays. Checked against 150 tracked plays with the tip position fitted
// per play: the ball lands 253 mm from where the camera saw it a second in and
// covers 95% of the real distance, with the cushion count right in the median.
// It is a drawing of which way a shot goes, not a prescription to aim by.

const SIM = (() => {
  const L = 2844, W = 1422, RADIUS = 30.75, DIAMETER = 61.5, G = 9810;
  // 이 셋은 영상에 맞춰 잡았다 (tools/fit_spin.py, 150개 플레이). 교과서 값은
  // 풀 당구천 기준이고 캐롬 대대는 더 빠르다 — 교과서 값으로는 공이 실제의
  // 72%밖에 못 가고 쿠션을 하나 덜 먹었다. src/physics/spin.py와 같은 값이어야
  // 하고 tests/test_sim_js.py가 그것을 지킨다.
  const SLIDING = 0.12, ROLLING = 0.006, SPIN_DECAY = 0.01;
  const CUSHION_SPEED = 60.0;
  // 쿠션에서 "회전을 적게 준" 것으로 칠 선. ⚠️ 재서 얻은 값이 아니라 고른 값이다.
  const LITTLE_SPIN = 0.15;
  const TIP_MM = 5, MAX_TIPS = 3;
  const RAIL_FRICTION = 0.08, RAIL_KEEPS_SIDE = 0.55, RAIL_KEEPS_SLIDE = 0.3;
  const REBOUND_IN = [0, 6.7, 17.8, 27.2, 38.0, 46.4, 56.0, 66.1, 79.3, 90.0];
  // 2026-09-25에 우리 캐롬 영상 6,115건으로 갈아 끼웠다. 2·3쿠션 오차가
  // 23%·31% 줄었다. 자세한 것과 남은 경고는 src/physics/spin.py.
  const REBOUND_OUT = [0.0, 25.6, 40.8, 46.1, 53.8, 58.6, 65.2, 71.8, 80.1, 90.0];
  const REBOUND_SPEED = [0.817, 0.817, 0.844, 0.829, 0.824, 0.818, 0.808, 0.835, 0.912, 0.912];
  // CUE_CARRY는 0이다. 수구가 충돌 뒤에도 앞으로 나아가는 것은 남은 구름이
  // 하는 일이지 상수가 하는 일이 아니다 — src/physics/spin.py의 설명 참조.
  const BALL_FRICTION = 0.06, BALL_RESTITUTION = 0.944, CUE_CARRY = 0.0;
  const REST = 12.0;
  const RAILS = { left: [1, 0], right: [-1, 0], top: [0, 1], bottom: [0, -1] };

  const interp = (x, xs, ys) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let i = 1;
    while (xs[i] < x) i++;
    const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + t * (ys[i] - ys[i - 1]);
  };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const speedOf = (b) => Math.hypot(b.v[0], b.v[1]);

  // A ball just struck. Where the tip lands sets the side and how much of the
  // slide is already gone: a ball struck high is part-way to rolling, one
  // struck low is spinning backwards under itself.
  function struck(position, heading, speed, tipsSide = 0, tipsVertical = 0) {
    const length = Math.hypot(heading[0], heading[1]) || 1;
    const v = [heading[0] / length * speed, heading[1] / length * speed];
    const sideways = clamp(tipsSide, -MAX_TIPS, MAX_TIPS) * TIP_MM;
    const vertical = clamp(tipsVertical, -MAX_TIPS, MAX_TIPS) * TIP_MM;
    // 오른쪽 회전은 위에서 보아 시계 방향이고, y가 아래로 가는 이 좌표계에서
    // 그것은 ω로 음수다. src/physics/spin.py와 같은 부호여야 한다.
    const side = -5 * speed * sideways / (2 * RADIUS * RADIUS);
    const rolled = clamp(vertical / (0.4 * RADIUS), -1.5, 1);
    return { p: position.slice(), v, side, slip: [v[0] * (1 - rolled), v[1] * (1 - rolled)] };
  }

  const still = (position) => ({ p: position.slice(), v: [0, 0], side: 0, slip: [0, 0] });

  function rollOn(ball, seconds) {
    ball.p = [ball.p[0] + ball.v[0] * seconds, ball.p[1] + ball.v[1] * seconds];
    const slipping = Math.hypot(ball.slip[0], ball.slip[1]);
    if (slipping > 1) {
      // Sliding: friction acts against the contact point, slowing the ball and
      // killing the slip seven halves as fast, which turns the slide into a
      // roll without anything having to decide when.
      const dir = [ball.slip[0] / slipping, ball.slip[1] / slipping];
      const lost = SLIDING * G * seconds;
      ball.v = [ball.v[0] - dir[0] * lost, ball.v[1] - dir[1] * lost];
      const left = Math.max(0, slipping - 3.5 * lost);
      ball.slip = [dir[0] * left, dir[1] * left];
    } else {
      const speed = speedOf(ball);
      if (speed > 1e-9) {
        const kept = Math.max(0, speed - ROLLING * G * seconds) / speed;
        ball.v = [ball.v[0] * kept, ball.v[1] * kept];
      }
      ball.slip = [0, 0];
    }
    const decay = SPIN_DECAY * G / RADIUS * seconds;
    ball.side = Math.abs(ball.side) <= decay ? 0 : ball.side - Math.sign(ball.side) * decay;
  }

  // The angle a ball comes off a rail with no side on it is measured; what the
  // side adds goes on top. The point touching the rail sits at -R along the
  // normal, so the slip along the rail is v - Rω: with that sign the wrong way
  // a cushion adds spin instead of eating it, and running side closes the
  // angle instead of opening it.
  function bounce(ball, rail) {
    const n = RAILS[rail], t = [-n[1], n[0]];
    const into = ball.v[0] * n[0] + ball.v[1] * n[1];
    let along = ball.v[0] * t[0] + ball.v[1] * t[1];
    if (into >= 0) return false;   // 이미 떠나는 중 — 쿠션을 먹은 것이 아니다
    // 이 아래로는 공이 쿠션에 기대고 있는 것이다. 튕기기는 하되 쿠션으로 세지
    // 않는다 — 세면 구석에 갇힌 공이 3쿠션을 "채워" 득점으로 둔갑한다.
    const counts = Math.hypot(ball.v[0], ball.v[1]) >= CUSHION_SPEED;
    // 들어갈 때 이 회전이 진행을 돕고 있었나(정회전) 거스르고 있었나(역회전).
    // 쿠션을 따라가는 방향과 회전이 만드는 표면 속도의 부호가 같으면 미끄럼
    // (v − Rω)이 줄어드니 정회전이다. 리버스는 1쿠션 역·2쿠션 정인 샷이라,
    // 쿠션 차례로는 갈리지 않고 이 값으로만 갈린다. spin.py와 같은 계산이다.
    // 부호만으로는 "회전을 적게 준" 경우를 못 잡는다 — 아주 작은 순회전도
    // running이 된다. 크기까지 본다. spin.py와 같은 계산·같은 선이어야 한다.
    const reach = Math.abs(along);
    const ratio = reach > 1e-6 ? RADIUS * ball.side * Math.sign(along) / reach : 0;
    ball.lastSpinRatio = ratio;
    ball.lastEnglish = Math.abs(ratio) < LITTLE_SPIN ? "none"
      : (ratio > 0 ? "running" : "reverse");
    const speed = Math.hypot(into, along);
    const incoming = Math.atan2(Math.abs(along), Math.abs(into)) * 180 / Math.PI;
    const outgoing = interp(incoming, REBOUND_IN, REBOUND_OUT);
    const kept = interp(incoming, REBOUND_IN, REBOUND_SPEED) * speed;
    const outNormal = kept * Math.cos(outgoing * Math.PI / 180);
    let outAlong = along ? Math.sign(along) * kept * Math.sin(outgoing * Math.PI / 180) : 0;

    const slip = outAlong - RADIUS * ball.side;
    const grip = RAIL_FRICTION * 1.7 * Math.abs(into);
    const change = -Math.sign(slip) * Math.min(grip, (2 / 7) * Math.abs(slip));
    outAlong += change;

    ball.v = [n[0] * outNormal + t[0] * outAlong, n[1] * outNormal + t[1] * outAlong];
    ball.side = (ball.side - (5 / (2 * RADIUS)) * change) * RAIL_KEEPS_SIDE;
    // A rail meets the ball above its equator, so it leaves mostly rolling.
    ball.slip = [ball.v[0] * RAIL_KEEPS_SLIDE, ball.v[1] * RAIL_KEEPS_SLIDE];
    return counts;
  }

  // Two balls do not part exactly along the line joining their centres: the
  // surfaces rub, and the cut angle and the side on the cue ball both show up
  // in how far the struck one is thrown off it.
  function collide(a, b) {
    const line = [b.p[0] - a.p[0], b.p[1] - a.p[1]];
    const gap = Math.hypot(line[0], line[1]);
    if (!gap) return;
    const n = [line[0] / gap, line[1] / gap], t = [-n[1], n[0]];
    const approach = (a.v[0] - b.v[0]) * n[0] + (a.v[1] - b.v[1]) * n[1];
    if (approach <= 0) return;

    const transfer = (1 + BALL_RESTITUTION) / 2 * approach;
    const aNormal = (a.v[0] * n[0] + a.v[1] * n[1]) - transfer + CUE_CARRY * Math.abs(approach);
    const bNormal = (b.v[0] * n[0] + b.v[1] * n[1]) + transfer;
    const aTangent = a.v[0] * t[0] + a.v[1] * t[1];
    const surface = aTangent + RADIUS * a.side;
    const thrown = -Math.sign(surface) * Math.min(BALL_FRICTION * transfer, Math.abs(surface) * 0.5);
    const bTangent = (b.v[0] * t[0] + b.v[1] * t[1]) - thrown;

    const wasRolling = [a.v[0] - a.slip[0], a.v[1] - a.slip[1]];
    a.v = [n[0] * aNormal + t[0] * (aTangent + thrown), n[1] * aNormal + t[1] * (aTangent + thrown)];
    b.v = [n[0] * bNormal + t[0] * bTangent, n[1] * bNormal + t[1] * bTangent];
    const passed = thrown * 5 / (2 * RADIUS);
    b.side += passed;
    a.side = a.side * 0.9 - passed * 0.2;
    // 분리각이 여기서 나온다. 두 공 사이의 충격은 중심을 잇는 선을 따라 두
    // 중심을 지나므로 수평축 회전에 토크를 주지 않는다 — 수구의 구름은 충돌을
    // 그대로 통과한다. src/physics/spin.py와 같은 식이어야 한다.
    a.slip = [a.v[0] - wasRolling[0], a.v[1] - wasRolling[1]];
    b.slip = b.v.slice();
  }

  function railReached(p) {
    if (p[0] < RADIUS) return "left";
    if (p[0] > L - RADIUS) return "right";
    if (p[1] < RADIUS) return "top";
    if (p[1] > W - RADIUS) return "bottom";
    return null;
  }

  // One stroke, in the 당점 a player can be given.
  // 20초는 파이썬 쪽 simulate_with_spin의 기본값과 같아야 한다. 천을 빠르게
  // 잡고 나니 공이 15초를 넘겨 굴러서, 두 벌이 서로 다른 자리에 공을 세웠다.
  function play(layout, cue, velocity, tipsSide = 0, tipsVertical = 0, maxSeconds = 20) {
    const colours = Object.keys(layout);
    const balls = {};
    for (const c of colours) balls[c] = still(layout[c]);
    const speed = Math.hypot(velocity[0], velocity[1]);
    balls[cue] = struck(layout[cue], velocity, speed, tipsSide, tipsVertical);

    const path = [balls[cue].p.slice()];
    const paths = {};
    for (const c of colours) paths[c] = [balls[c].p.slice()];
    const events = [];
    const frame = 1 / 60;
    let clock = 0, nextFrame = frame;

    while (clock < maxSeconds) {
      let fastest = 0;
      for (const c of colours) fastest = Math.max(fastest, speedOf(balls[c]));
      if (fastest < REST) break;

      let clearance = 1e9;
      for (const c of colours) {
        const p = balls[c].p;
        clearance = Math.min(clearance, p[0] - RADIUS, L - RADIUS - p[0],
                             p[1] - RADIUS, W - RADIUS - p[1]);
        for (const other of colours) {
          if (other === c) continue;
          clearance = Math.min(clearance,
            Math.hypot(p[0] - balls[other].p[0], p[1] - balls[other].p[1]) - DIAMETER);
        }
      }
      const step = Math.min(Math.max(clearance * 0.5, 4) / fastest, frame);

      for (const c of colours) rollOn(balls[c], step);

      for (const c of colours) {
        const rail = railReached(balls[c].p);
        if (!rail) continue;
        balls[c].p = [clamp(balls[c].p[0], RADIUS, L - RADIUS),
                      clamp(balls[c].p[1], RADIUS, W - RADIUS)];
        // 되튕긴 경우에만 쿠션으로 친다. 쿠션을 따라 기어가는 공은 매 걸음
        // 여기로 돌아오는데, 그때마다 한 개씩 세면 한 샷에 쿠션이 370개가
        // 나오고 — judge()가 쿠션을 세므로 — 세 개를 못 넘긴 샷이 득점으로
        // 둔갑한다.
        if (bounce(balls[c], rail) && c === cue) {
          events.push({ kind: "cushion", detail: rail, at: clock,
                        p: balls[c].p.slice(), english: balls[c].lastEnglish });
        }
      }

      for (let i = 0; i < colours.length; i++) {
        for (let j = i + 1; j < colours.length; j++) {
          const a = balls[colours[i]], b = balls[colours[j]];
          const gap = Math.hypot(a.p[0] - b.p[0], a.p[1] - b.p[1]);
          if (gap > DIAMETER || gap === 0) continue;
          const push = (DIAMETER - gap) / 2 + 0.01;
          const dir = [(b.p[0] - a.p[0]) / gap, (b.p[1] - a.p[1]) / gap];
          a.p = [a.p[0] - dir[0] * push, a.p[1] - dir[1] * push];
          b.p = [b.p[0] + dir[0] * push, b.p[1] + dir[1] * push];
          const closing = (a.v[0] - b.v[0]) * dir[0] + (a.v[1] - b.v[1]) * dir[1];
          if (closing > 0) collide(a, b); else collide(b, a);
          if (colours[i] === cue || colours[j] === cue) {
            const other = colours[i] === cue ? colours[j] : colours[i];
            events.push({ kind: "ball", detail: other, at: clock, p: balls[cue].p.slice() });
          } else {
            // 적구끼리 부딪힌 것. 수구가 2적구에 닿기 전이면 키스다.
            events.push({ kind: "kiss", detail: `${colours[i]}-${colours[j]}`, at: clock, p: a.p.slice() });
          }
        }
      }

      clock += step;
      if (clock >= nextFrame) {
        path.push(balls[cue].p.slice());
        for (const c of colours) paths[c].push(balls[c].p.slice());
        nextFrame += frame;
      }
    }
    const rest = {};
    for (const c of colours) rest[c] = balls[c].p.slice();
    return { path, paths, rest, events, seconds: clock };
  }

  // Did it carom: first object ball, then three cushions before the other one.
  function judge(shot) {
    const balls = shot.events.filter((e) => e.kind === "ball");
    if (!balls.length) return { scored: false };
    const first = balls[0];
    const second = balls.find((e) => e.detail !== first.detail);
    if (!second) return { scored: false, first: first.detail };
    const rails = shot.events
      .filter((e) => e.kind === "cushion" && e.at < second.at)
      .map((e) => e.detail);
    return { scored: rails.length >= 3, first: first.detail, second: second.detail, rails };
  }

  // 스트로크 강도 — 초속 대신 선수가 쓸 수 있는 자.
  //   강도 1 = 수구가 장축(2844 mm)만큼 굴러가는 세기, 강도 n = 그 n 배.
  // 기준 샷은 짧은 쿠션 한가운데에서 긴 쿠션과 나란히 무회전으로 곧게 친 수구.
  // 표는 src/physics/strength.py와 같은 값이고 tests/test_strength.py가 둘을
  // 붙들어 둔다 — 물리가 바뀌면 같은 이름이 다른 세기를 가리키게 되므로.
  const STRENGTH = [0.14, 0.32, 0.57, 0.99, 1.30, 1.68, 2.20, 2.70, 3.24, 3.78, 4.17, 4.73, 5.31, 6.20, 6.89, 7.67, 8.23, 9.12, 10.73, 12.08];
  const STRENGTH_SPEED = [300, 450, 600, 800, 1000, 1200, 1500, 1800, 2200, 2600, 3000, 3500, 4000, 4800, 5600, 6800, 8000, 10000, 14000, 20000];

  function between(x, from, to) {
    if (x <= from[0]) return to[0] * x / from[0];
    for (let i = 1; i < from.length; i++) {
      if (x <= from[i]) {
        const t = (x - from[i - 1]) / (from[i] - from[i - 1]);
        return to[i - 1] + t * (to[i] - to[i - 1]);
      }
    }
    return to[to.length - 1];
  }

  const speedFor = (strength) => between(strength, STRENGTH, STRENGTH_SPEED);
  const strengthOf = (speed) => between(speed, STRENGTH_SPEED, STRENGTH);

  return { play, judge, struck, speedFor, strengthOf,
           L, W, RADIUS, DIAMETER, MAX_TIPS, TIP_MM, LONG_RAIL: L };
})();
