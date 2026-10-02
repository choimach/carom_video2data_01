// 오프라인에서도 열리게. 페이지와 스크립트는 **네트워크 먼저** — 새 판을 올리면 바로 받고,
// 네트워크가 없을 때만 저장해 둔 것을 쓴다 (당구장 지하에서도 돈다).
const CACHE = "carom-board-v1";
self.addEventListener("install", (e) => { self.skipWaiting(); });
self.addEventListener("activate", (e) => { e.waitUntil(self.clients.claim()); });
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;
  e.respondWith(
    fetch(e.request).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
      return res;
    }).catch(() => caches.match(e.request).then((hit) => hit || caches.match("./index.html")))
  );
});
