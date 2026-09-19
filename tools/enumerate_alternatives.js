// 프로가 고른 길 옆에, 고를 수 있었는데 버린 길을 세워 둔다.
//
// 이 프로젝트가 되풀이해 부딪힌 벽: 영상은 선수가 **고른 것**만 남기고, 같은
// 배치에서 가능했는데 치지 않은 길은 흔적을 남기지 않는다. 그래서 순위는 "이
// 배치와 닮은 자리에서 이 유형을 몇 명이 골랐나"까지만 배울 수 있었다. 무엇
// **대신** 골랐는지는 배울 수 없었다.
//
// 그런데 그 잃어버린 절반은 만들 수 있다. 배치는 남아 있으니, 그 배치에 조언판
// 과 똑같은 탐색을 돌리면 그날 테이블 위에 실제로 있었던 길이 전부 나온다.
// 그중 하나가 프로가 친 것이고, 나머지가 버린 것이다.
//
//   node tools/enumerate_alternatives.js              # 전부, 이어받기 됨
//   node tools/enumerate_alternatives.js --limit 50
//
// data/alternatives.jsonl에 한 줄에 한 플레이씩 붙여 쓴다. 중간에 끊겨도 이미
// 한 것은 건너뛴다 — 한 시간짜리 작업이라 처음부터 다시 하지 않는 편이 낫다.

const fs = require('fs');
const path = require('path');

const ROOT = path.dirname(__dirname);
const SIM = eval(fs.readFileSync(path.join(ROOT, 'build', 'sim.js'), 'utf8') + '\nSIM;');

const ORDER = ['white', 'yellow', 'red'];
const SHORT = new Set(['left', 'right']);
const FINE = 0.25;
const STRENGTHS = [3.5, 4.5, 5.5, 7.0];
const SPEEDS = STRENGTHS.map((s) => Math.round(SIM.speedFor(s)));
const CLOCK = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const atClock = (hour, tips) => {
  const turn = (hour % 12) / 12 * 2 * Math.PI;
  return [tips * Math.sin(turn), tips * Math.cos(turn)];
};
const TIP_POINTS = [[0, 0], ...CLOCK.map((h) => atClock(h, 2))];

const thicknessSteps = (reach) => {
  const span = 2 * Math.asin(Math.min(1, SIM.DIAMETER / Math.max(reach, SIM.DIAMETER)))
    * 180 / Math.PI;
  return Math.max(8, Math.min(30, Math.round(span / 0.8)));
};

// 조언판과 같은 판정이어야 한다. 다르면 여기서 만든 "버린 길"이 조언판이
// 실제로 내놓는 후보와 다른 것이 되어, 배운 것을 쓸 자리가 없다.
function kissed(shot) {
  const balls = shot.events.filter((e) => e.kind === 'ball');
  if (!balls.length) return false;
  const first = balls[0];
  const second = balls.find((e) => e.detail !== first.detail);
  if (!second) return false;
  if (shot.events.some((e) => e.kind === 'kiss' && e.at <= second.at)) return true;
  return balls.some((e) => e !== first && e.detail === first.detail && e.at < second.at);
}

function nameRoute(shot, judged, aim, face) {
  const balls = shot.events.filter((e) => e.kind === 'ball');
  const first = balls[0];
  const before = shot.events.filter((e) => e.kind === 'cushion' && e.at < first.at);
  if (before.length >= 2) return '뱅크샷';
  if (before.length === 1) return '걸어치기';
  const second = balls.find((e) => e.detail !== first.detail);
  const rails = shot.events
    .filter((e) => e.kind === 'cushion' && e.at > first.at && e.at < second.at)
    .map((e) => e.detail);
  if (rails.length < 3) return null;
  if (rails.length >= 5) return '대회전';

  const path = shot.paths[shot.cue] || [];
  let area = 0;
  for (let i = 0; i + 1 < path.length; i++) {
    const ax = path[i][0] - SIM.L / 2, ay = path[i][1] - SIM.W / 2;
    const bx = path[i + 1][0] - SIM.L / 2, by = path[i + 1][1] - SIM.W / 2;
    area += ax * by - ay * bx;
  }
  const circuitRight = area > 0;
  const sameHand = (face === 'right') === circuitRight;
  const shortFirst = SHORT.has(rails[0]);
  if (sameHand) return shortFirst ? '앞돌리기' : '옆돌리기';
  return shortFirst ? '빗겨치기' : '뒤돌리기';
}

function search(layout, cue) {
  const from = layout[cue];
  const jobs = [];
  for (const colour of ORDER) {
    if (colour === cue || !layout[colour]) continue;
    const to = layout[colour];
    const straight = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI;
    const reach = Math.hypot(to[0] - from[0], to[1] - from[1]);
    if (reach < 70) continue;
    const steps = thicknessSteps(reach);
    for (let i = 0; i <= steps; i++) {
      const offset = (i / steps) * SIM.RADIUS * 2;
      const swing = Math.asin(Math.min(1, offset / reach)) * 180 / Math.PI;
      for (const hand of (i === 0 ? [0] : [1, -1])) {
        const deg = ((straight + hand * swing) % 360 + 360) % 360;
        for (const speed of SPEEDS) jobs.push([deg, speed, 0, 0]);
      }
    }
  }

  // 두께 사다리는 적구를 똑바로 겨냥하므로, **쿠션을 먼저 맞는** 길 — 걸어치기와
  // 뱅크샷 — 은 우연히 아니면 걸리지 않는다. 실제로 걸어치기는 0/7, 뱅크샷은
  // 5/11이었다. 그대로 배우면 "프로는 걸어치기를 안 친다"를 배우는데, 그건
  // 프로에 대한 사실이 아니라 이 탐색에 대한 사실이다.
  //
  // 거울상을 겨냥하는 방법은 쓰지 않는다: 쿠션은 거울이 아니다 (측정된 반사
  // 곡선은 17.8도로 들어간 공을 33.7도로 내보낸다). 대신 전방위를 2도로 한 번
  // 훑는다 — 무엇을 먼저 맞든 상관하지 않으므로 이쪽이 맞는 그물이고, 값도
  // 싸다 (180각 × 4세기 = 720번, 0.5초 남짓).
  for (const speed of SPEEDS) {
    for (let deg = 0; deg < 360; deg += 2) jobs.push([deg, speed, 0, 0]);
  }

  const hits = [], promising = [];
  const go = (deg, speed, side, up) => {
    const rad = deg * Math.PI / 180;
    const aim = [Math.cos(rad), Math.sin(rad)];
    const shot = SIM.play(layout, cue, [aim[0] * speed, aim[1] * speed], side, up);
    shot.cue = cue;
    const judged = SIM.judge(shot);
    const balls = shot.events.filter((e) => e.kind === 'ball');
    const cushions = shot.events.filter((e) => e.kind === 'cushion').length;
    const maybe = balls.length > 0 && cushions >= 2;
    if (!judged.scored || kissed(shot)) return maybe;
    const target = layout[judged.first];
    const across = aim[0] * (target[1] - from[1]) - aim[1] * (target[0] - from[0]);
    const face = across > 0 ? 'left' : 'right';
    const route = nameRoute(shot, judged, aim, face);
    if (!route) return maybe;
    const way = shot.paths[judged.first] || [];
    let pushed = 0;
    for (let i = 1; i < way.length; i++) {
      pushed += Math.hypot(way[i][0] - way[i - 1][0], way[i][1] - way[i - 1][1]);
    }
    hits.push({
      route, first: judged.first, face,
      thickness: Math.max(0, Math.min(1, 1 - Math.abs(across) / SIM.DIAMETER)),
      side, up, deg, strength: SIM.strengthOf(speed),
      rails: judged.rails.length, pushed,
    });
    return true;
  };

  for (const job of jobs) if (go(...job)) promising.push(job);
  const key4 = (j) => `${j[0]}:${j[1]}:${j[2]}:${j[3]}`;
  const seen = new Set(jobs.map(key4));
  for (const [deg, speed] of promising) {
    for (const [side, up] of TIP_POINTS) {
      for (let d = -1.0; d <= 1.0; d += FINE) {
        const at = Math.round((((deg + d) % 360 + 360) % 360) / FINE) * FINE;
        const job = [at, speed, side, up];
        if (seen.has(key4(job))) continue;
        seen.add(key4(job));
        go(...job);
      }
    }
  }
  return hits;
}

// 한 배치의 답은 수백 줄이지만, 선수가 고르는 단위는 "어느 공의 어느 면으로
// 무슨 유형" 하나다. 그 단위로 묶고, 묶음마다 가장 여유 있는 줄로 대표한다.
function bucket(hits) {
  const by = new Map();
  for (const hit of hits) {
    const key = `${hit.route}|${hit.first}|${hit.face}`;
    if (!by.has(key)) by.set(key, []);
    by.get(key).push(hit);
  }
  const out = [];
  for (const [key, list] of by) {
    // 각도가 이어진 폭이 곧 조준 여유다. 같은 묶음 안에서 가장 굵은 줄을 고른다.
    const degrees = [...new Set(list.map((h) => Math.round(h.deg / FINE) * FINE))].sort((a, b) => a - b);
    let widest = FINE, run = FINE;
    for (let i = 1; i < degrees.length; i++) {
      run = (degrees[i] - degrees[i - 1] <= FINE + 1e-9) ? run + FINE : FINE;
      if (run > widest) widest = run;
    }
    const middle = (field) => {
      const s = list.map((h) => h[field]).sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)];
    };
    const [route, first, face] = key.split('|');
    out.push({
      key, route, first, face,
      lines: list.length,
      room: Math.round(widest * 100) / 100,
      thickness: Math.round(middle('thickness') * 1000) / 1000,
      strength: Math.round(middle('strength') * 10) / 10,
      rails: middle('rails'),
      pushed: Math.round(middle('pushed')),
      side: Math.round(middle('side') * 100) / 100,
      up: Math.round(middle('up') * 100) / 100,
    });
  }
  return out.sort((a, b) => b.room - a.room);
}

function main() {
  const args = process.argv.slice(2);
  const limit = args.includes('--limit') ? Number(args[args.indexOf('--limit') + 1]) : Infinity;
  // 고치려는 유형만 따로 재 볼 때 쓴다. 전체를 돌리며 기다리면 그 유형이
  // 표본에 몇 개 안 들어와서 무엇이 나아졌는지 보이지 않는다.
  const only = args.includes('--route') ? args[args.indexOf('--route') + 1] : null;
  const model = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'model.json'), 'utf8'));
  const rows = Array.isArray(model) ? model : (model.plays || model.rows);
  const out = path.join(ROOT, 'data', 'alternatives.jsonl');

  const done = new Set();
  // 유형만 재 볼 때는 장부에 쓰지 않으므로, 이미 한 것도 다시 본다.
  if (!only && fs.existsSync(out)) {
    for (const line of fs.readFileSync(out, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { done.add(JSON.parse(line).id); } catch (e) { /* 끊긴 줄은 버린다 */ }
    }
  }
  console.log(`이미 한 것 ${done.size}개`);

  const usable = rows.filter((r) => (!only || r.route === only))
    .filter((r) => r.route && r.first_object_ball
    && r.struck_side !== null && r.struck_side !== undefined && r.layout_mm
    && ORDER.every((c) => r.layout_mm[c]));
  console.log(`대상 ${usable.length}개 (배치·유형·면이 다 있는 플레이)\n`);

  let counted = 0, began = Date.now();
  for (const row of usable) {
    const id = `${row.match}:${row.inning}:${row.shot}`;
    if (done.has(id)) continue;
    if (counted >= limit) break;

    const layout = {};
    for (const c of ORDER) layout[c] = row.layout_mm[c].map(Number);
    const hits = search(layout, row.cue);
    const buckets = bucket(hits);
    // 프로가 실제로 친 것. struck_side 양수가 왼쪽 면 — app_data.py와 같은 규약.
    const chose = `${row.route}|${row.first_object_ball}|${row.struck_side > 0 ? 'left' : 'right'}`;
    const record = {
      id, match: row.match, split: row.split, scored: row.scored,
      cue: row.cue, layout: layout,
      chose,
      // 그날 실제로 있었던 길 전부. 이 중 하나가 chose이고, 나머지가 버린 것이다.
      found: buckets,
      // 탐색이 프로가 친 길을 찾아냈는가. 못 찾았으면 그 플레이는 "버린 길"의
      // 근거로 쓸 수 없다 — 우리 탐색이 놓친 것이지 선수가 버린 것이 아니다.
      reached: buckets.some((b) => b.key === chose),
      pro: {
        thickness: row.thickness, strength: row.speed_ms
          ? Math.round(SIM.strengthOf(row.speed_ms * 1000) * 10) / 10 : null,
      },
    };
    if (!only) fs.appendFileSync(out, JSON.stringify(record) + '\n');
    else console.log(`  ${record.reached ? '찾음 ' : '놓침 '} ${record.chose}  갈래 ${record.found.length}개`);
    counted++;
    if (counted % 10 === 0) {
      const each = (Date.now() - began) / counted / 1000;
      const left = (usable.length - done.size - counted) * each / 60;
      process.stdout.write(`  ${counted}개 · 배치당 ${each.toFixed(1)}초 · 남은 시간 ${left.toFixed(0)}분\n`);
    }
  }
  console.log(`\n${counted}개 새로 담았습니다 -> ${out}`);
}

main();
