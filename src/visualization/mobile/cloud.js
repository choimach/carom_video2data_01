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
import { getFirestore, collection, addDoc, serverTimestamp }
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
const auth = getAuth(app), db = getFirestore(app);
let signing = null;
const uid = async () => {
  if (auth.currentUser) return auth.currentUser.uid;
  signing = signing || signInAnonymously(auth);
  return (await signing).user.uid;
};

window.CAROM_CLOUD = {
  build: window.CAROM_BUILD || null,
  async send(record, text) {
    const me = await uid();
    await addDoc(collection(db, "users", me, "feedback"), {
      userId: me, schemaVersion: 1, app: "board", build: window.CAROM_BUILD || null,
      record, text, createdAt: serverTimestamp(),
    });
    return true;
  },
};

// 버튼 이름을 하는 일에 맞춘다.
const button = document.getElementById("copy");
if (button) button.textContent = "결과 보내기";
uid().catch(() => {});   // 미리 로그인해 둔다 — 첫 보내기가 빨라진다

if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
