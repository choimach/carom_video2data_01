// 폰 앱(Firebase Hosting에 올린 판)에서만 붙는다 — 조언판의 "결과 복사"가 만든 기록을
// Firestore에도 보낸다. 조언판(claude.ai)에는 이 파일이 없다.
//
// 프로젝트 carom001 (선수의 구글 계정). 같은 프로젝트에 2026-09-10의 폰 앱
// (carom_bot_01, https://carom001.web.app)이 있다 — 그 앱의 기록(users/{uid}/cycles)은
// 건드리지 않고 users/{uid}/feedback 에만 쓴다. 규칙은 carom_bot_01/firestore.rules.
// 익명 로그인: 기기마다 uid가 다르고, 자기 기록만 읽고 쓴다.
//
// 가져오기: tools/pull_feedback.py → data/table_feedback.json
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import { getAuth, signInAnonymously } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";
import { initializeFirestore, persistentLocalCache, persistentSingleTabManager,
         collection, doc, setDoc, serverTimestamp, waitForPendingWrites }
  from "https://www.gstatic.com/firebasejs/10.14.1/firebase-firestore.js";

// 웹 앱 설정 — 공개되는 값이다 (접근은 Firestore 규칙이 막는다).
const app = initializeApp({
  projectId: "carom001",
  appId: "1:439190673476:web:98c0a0bb61d6c6685d8411",
  apiKey: "AIzaSyBJonAS5xuqGKN_fDHXmhPi_rv6xtP4-do",
  authDomain: "carom001.firebaseapp.com",
  storageBucket: "carom001.firebasestorage.app",
  messagingSenderId: "439190673476",
});
const auth = getAuth(app);
// ★폰 안에 먼저 쓴다 (2026-10-03). 당구장 지하처럼 신호가 없는 곳에서 보내면 기록이 앱
// 메모리에만 있다가 앱을 닫으면 사라졌다. 이 캐시는 IndexedDB에 남고, 다음에 연결되면
// Firestore가 알아서 올린다 — 앱을 닫았다 열어도.
const db = initializeFirestore(app, {
  localCache: persistentLocalCache({ tabManager: persistentSingleTabManager() }),
});
let signing = null;
const uid = async () => {
  await auth.authStateReady();          // 한 번 로그인한 기기는 오프라인에서도 uid가 살아 있다
  if (auth.currentUser) return auth.currentUser.uid;
  signing = signing || signInAnonymously(auth);
  return (await signing).user.uid;
};

window.CAROM_CLOUD = {
  build: window.CAROM_BUILD || null,
  // 돌려주는 값: "sent"(서버가 받음) · "queued"(폰에 저장, 연결되면 올라감).
  // photos: 사진을 함께 보낼 때 [data URL(JPEG)…] — 장마다 users/{uid}/photos 문서 하나 (2026-10-04).
  async send(record, text, photos = []) {
    const me = await uid();
    // Firestore는 undefined 값이 하나라도 있으면 문서 전체를 거절한다 (2026-10-04: 사진 요약의 빈 점수 하나로
    // 보내기가 통째로 실패했다). JSON을 한 번 거치면 undefined가 빠진다.
    record = JSON.parse(JSON.stringify(record));
    const ref = doc(collection(db, "users", me, "feedback"));   // id를 여기서 정한다 — 다시 보내도 하나
    const writes = photos.map((jpeg, slot) => setDoc(doc(collection(db, "users", me, "photos")), {
      userId: me, schemaVersion: 1, feedbackId: ref.id, slot, at: record.at || null, jpeg, createdAt: serverTimestamp(),
    }));
    writes.push(setDoc(ref, {
      userId: me, schemaVersion: 1, app: "board", build: window.CAROM_BUILD || null,
      record, text, photos: photos.length, createdAt: serverTimestamp(),
    }));
    const written = Promise.all(writes);
    // 서버 응답을 4초까지만 기다린다. 그 안에 안 오면 폰에 저장된 채 연결을 기다린다.
    const late = new Promise((done) => setTimeout(() => done("queued"), 4000));
    return Promise.race([written.then(() => "sent"), late]);
  },
};

// 버튼 이름을 하는 일에 맞춘다.
const button = document.getElementById("copy");
if (button) button.textContent = "결과 보내기";
// 미리 로그인해 두고, **폰에 남아 있던 기록을 올린다.** Firestore는 처음 쓰일 때까지 시작하지
// 않아서, 오프라인에서 저장한 기록이 앱을 다시 열어도 다음 "보내기"를 누를 때까지 폰에 있었다
// (2026-10-03, 헤드리스로 확인: 다시 연 뒤 25초 동안 그대로). 여기서 시작시킨다.
uid().then(() => waitForPendingWrites(db)).catch(() => {});

// 어느 판이 떠 있는지 화면 맨 아래에 적는다 — 선수: "자동으로 update되나?" (2026-10-03).
// 서비스 워커가 네트워크 먼저라 온라인에서 열면 늘 최신이지만, 눈으로 확인할 수 있어야 한다.
const stamp = document.createElement("p");
stamp.textContent = `폰 앱 · 빌드 ${window.CAROM_BUILD || "?"}`;
stamp.style.cssText = "margin:10px 16px 18px;font-size:11px;color:#8b968e;text-align:center";
document.body.appendChild(stamp);

if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
