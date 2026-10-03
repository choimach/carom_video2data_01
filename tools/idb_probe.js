// 페이지 안에서: firestore IndexedDB의 mutations 가게에 몇 개가 있나
window.__probe = () => new Promise((done) => {
  const req = indexedDB.open('firestore/[DEFAULT]/carom001/main');
  req.onsuccess = () => {
    const db = req.result;
    const names = [...db.objectStoreNames];
    if (!names.includes('mutations')) return done({ names });
    const tx = db.transaction(['mutations', 'owner'], 'readonly');
    const m = tx.objectStore('mutations').getAll();
    const o = tx.objectStore('owner').getAll();
    tx.oncomplete = () => { done({ mutations: m.result.length, owner: o.result }); db.close(); };
  };
  req.onerror = () => done({ error: String(req.error) });
});
