"""Screen a list of candidate VODs before downloading any of them in full."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_acquisition.screen import screen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
listing = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tools", "pick10.txt")
rows = [l.rstrip("\n").split("|", 2) for l in open(listing, encoding="utf-8", errors="replace") if "|" in l]
# Results accumulate as each match is judged, so a stopped run resumes rather
# than paying for the same previews twice.
cache = os.path.join(ROOT, "data", "_screening.json")
out = json.load(open(cache)) if os.path.exists(cache) else []
done = {r.get("vod_id") for r in out}
began = time.time()
for vid, ev, title in rows:
    if vid in done:
        print(f"  seen {ev}", flush=True)
        continue
    # 한 경기의 서버 오류(502 등)로 목록 전체가 멈추지 않게 (2026-10-09 — 첫 경기에서 죽어 뒤의 받기까지 헛돌았다).
    # 실패한 것은 장부에 적지 않으므로 다음 실행에서 다시 본다.
    try:
        result = screen(f"https://vod.sooplive.com/player/{vid}",
                        workdir=os.path.join(ROOT, "data", "_screen"))
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED {ev} {vid}: {error}", flush=True)
        continue
    result["event"], result["vod_id"] = ev, vid
    out.append(result)
    json.dump(out, open(os.path.join(ROOT, "data", "_screening.json"), "w"),
              indent=1, ensure_ascii=False)
good = [r for r in out if r.get("usable")]
print(f"\n{len(good)}/{len(out)} usable, screened in {(time.time() - began) / 60:.0f} min", flush=True)
for r in good:
    print(f"   {r['event']:9s} {r['vod_id']}  err {r['median_error_mm']:.1f} mm  "
          f"scale {r['mm_per_px']:.4f}  table in {r['calibrated_share'] * 100:.0f}% of samples", flush=True)
