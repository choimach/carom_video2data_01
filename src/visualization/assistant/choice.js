// 프로가 같은 배치에서 무엇 **대신** 무엇을 골랐는지로 맞춘 점수 — 한 벌.
//
// `tools/learn_choices.py`가 프로 배치 1,580개에서 뽑았다. 그 배치마다 그날
// 실제로 있었던 길을 전부 세우고 (버린 길 30,167개), 고른 것과 버린 것을 가르는
// 것이 무엇인지 맞춘 것이다. 영상만으로는 얻을 수 없는 절반이다.
//
// 경기 단위 10겹 교차검증 (1,580판 전부를 한 번씩 test):
//
//   아무렇게나   1등  4% · 3등 안 15% · 자리 중앙값 10.0
//   손 식       1등 19% · 3등 안 42% · 자리 중앙값  5.0
//   배운 식      1등 25% · 3등 안 50% · 자리 중앙값  4.0
//
// 짝지은 부호검정 5.7 표준편차. ⚠️ 다만 비교에 쓴 손 식은 **기하 부분만**이다 —
// 실제 조언판 식에 있는 이웃 프로 수와 득점률은 이 자료에 없어 뺐다. 그래서
// 화면에서는 둘을 **곱해** 쓴다.
//
// 특징의 차례와 모양은 learn_choices.py의 features()와 **반드시 같아야** 한다.
// tests/test_choice_js.py가 두 쪽에 같은 값을 물어 대조한다.

const CHOICE = (() => {
  // learn_choices.py의 NAMES와 같은 차례.
  const NAMES = ["여유", "두께", "세기", "쿠션수", "1적구이동",
                 "줄두께", "회전량", "상단당점", "이웃프로수", "이웃득점률", "기준"];

  // room      : 조준 여유(도) — 좌우로 얼마나 빗나가도 득점하는가
  // thickness : 두께 0~1
  // strength  : 스트로크 강도 (장축 몇 개)
  // rails     : 2적구 전 쿠션 수
  // pushed    : 1적구가 굴러간 거리(mm)
  // lines     : 이 공략에 들어가는 줄의 수 — 얼마나 너그러운가
  // side, up  : 당점 (팁)
  // chosen : 닮은 배치의 프로 40명 중 몇 명이 이 길을 골랐나
  // rate   : 그중 몇이 넣었나 (한 번의 성공과 한 번의 실패를 미리 얹어 누른 값)
  function features({ room, thickness, strength, rails, pushed, lines, side, up,
                      chosen, rate }) {
    const tips = Math.hypot(side || 0, up || 0);
    return [
      Math.log1p(room),
      thickness,
      strength / 7,
      Math.min(rails, 6) / 6,
      Math.log1p(pushed) / 10,
      Math.log1p(lines || 1) / 5,
      tips / 3,
      (up || 0) > 0.3 ? 1 : 0,
      // 2026-09-23에 한 번 뺐다가 되돌렸다. 뺄 때 잰 것은 "이웃이 무엇을 고를지
      // 맞힐 수 있나"였고 그건 거의 안 된다. 그런데 쓸모는 다른 데 있었다 —
      // **후보의 45%는 이웃 중 아무도 고른 적이 없고**, 우승자를 맞히지 않아도
      // 그런 길을 가라앉히는 것만으로 순위가 좋아진다. 1등 24.7% → 33.4%,
      // 3등 안 49.9% → 60.0%, 짝지은 부호검정 10.8 표준편차.
      // 선수가 먼저 알아챘다: "프로 0명이라면서 왜 이 공을 최우선으로?"
      Math.log1p(chosen || 0),
      rate == null ? 0.5 : rate,
      1,
    ];
  }

  // 곱해서 쓰려면 양수여야 한다. 순위만 보는 것이라 밑이 무엇이든 차례는 같다.
  function score(branch, weights) {
    const x = features(branch);
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * weights[i];
    return Math.exp(sum);
  }

  return { NAMES, features, score };
})();

if (typeof module !== "undefined") module.exports = CHOICE;
