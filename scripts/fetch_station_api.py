# 전국 공영자전거 대여소 정보 API(inf_101_00010001_v2) -> 세종 대여소 좌표 CSV
import json, os, sys, time, urllib.request, urllib.parse, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "sejong")
key = open(os.path.join(OUT, "api_key.txt"), encoding="utf-8-sig").read().strip()
BASES = ["https://apis.data.go.kr/B551982/pbdo_v2", "https://apis.data.go.kr/B551982/pbdo"]
OP = "/inf_101_00010001_v2"

def get(base, page, n=1000):
    q = urllib.parse.urlencode({"pageNo": page, "numOfRows": n, "type": "json"})
    url = f"{base}{OP}?serviceKey={key}&{q}"   # Encoding 키는 그대로 붙임
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")

def items_of(js):
    # 응답 구조가 문서마다 달라서 'item' 리스트를 재귀로 찾음
    if isinstance(js, dict):
        for k, v in js.items():
            if k.lower() in ("item", "items") and isinstance(v, list): return v
            r = items_of(v)
            if r is not None: return r
    if isinstance(js, list) and js and isinstance(js[0], dict) and len(js[0]) > 3: return js
    return None

base = None
for b in BASES:
    try:
        txt = get(b, 1, 10)
        open(os.path.join(OUT, "api_probe.txt"), "w", encoding="utf-8").write(txt[:5000])
        if txt.strip().startswith("{"):
            base = b; break
        print("응답이 JSON이 아님:", b, txt[:200])
    except Exception as e:
        print("실패:", b, e)
if base is None:
    sys.exit("API 호출 실패 - data/raw/sejong/api_probe.txt 내용을 확인해줘")

rows, page = [], 1
while True:
    js = json.loads(get(base, page))
    it = items_of(js) or []
    rows += it
    print(f"page {page}: {len(it)}건 (누적 {len(rows)})")
    if len(it) < 1000: break
    page += 1; time.sleep(0.3)

with open(os.path.join(OUT, "station_api_all.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
cols = sorted({k for r in rows for k in r})
def is_sejong(r):
    s = json.dumps(r, ensure_ascii=False)
    return ("세종" in s) or ("SJ_" in s) or ("SJ-" in s)
sj = [r for r in rows if is_sejong(r)]
with open(os.path.join(OUT, "sejong_station_api.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sj)
print("전체", len(rows), "건 / 세종", len(sj), "건 저장 완료")
print("컬럼:", cols)
