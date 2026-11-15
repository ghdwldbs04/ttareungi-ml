# 세종 어울링 건별 데이터(2025년 4개 분기) -> data/processed/sejong/ (다른 도시와 같은 형식)
import json, re, zipfile, io, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "sejong")
OUT = os.path.join(ROOT, "data", "processed", "sejong"); os.makedirs(OUT, exist_ok=True)
YEAR = 2025
MIN_ACTIVE_DAYS = 60

def norm(s):
    m = re.match(r"^(SJ|SN)[_-]?0*(\d+)$", str(s).strip())
    return f"SJ-{int(m.group(2)):05d}" if m else None

# 1) 대여소 좌표 (API)
api = pd.read_csv(os.path.join(RAW, "sejong_station_api.csv"), dtype=str, encoding="utf-8-sig")
api = api[api.lcgvmnInstNm.str.startswith("세종", na=False)].copy()
api["station_id"] = api.rntstnId.map(norm)
api = api.dropna(subset=["station_id"]).drop_duplicates("station_id")
st = pd.DataFrame({"station_id": api.station_id, "station_name": api.rntstnNm,
                   "lat": api.lat.astype(float), "lon": api.lot.astype(float)})
st.to_csv(os.path.join(OUT, "stations.csv"), index=False, encoding="utf-8-sig")
print("대여소 좌표", len(st))

# 2) 건별 데이터 (2025년 분기 파일만)
z = zipfile.ZipFile(os.path.join(RAW, "세종도시교통공사_공공자전거(어울링) 이용 현황_20251231.zip"))
parts, q = [], {"raw": 0, "bad_datetime": 0, "out_of_year": 0, "short_same_station": 0, "unknown_station": 0}
for info in z.infolist():
    name = info.filename
    try: name = name.encode("cp437").decode("cp949")
    except Exception: pass
    if f"{YEAR}년" not in name: continue
    raw = z.read(info)
    for enc in ("utf-8-sig", "cp949"):
        try: df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str); break
        except UnicodeDecodeError: continue
    df.columns = [c.strip() for c in df.columns]
    print(name, len(df))
    q["raw"] += len(df)
    d = pd.DataFrame({
        "station": df["시작 대여소"].map(norm), "ret_station": df["반납 대여소"].map(norm),
        "rt": pd.to_datetime(df["대여시간"], errors="coerce", format="mixed"),
        "rr": pd.to_datetime(df["반납시간"], errors="coerce", format="mixed"),
        "dist": pd.to_numeric(df["주행거리"], errors="coerce")})
    bad = d.rt.isna(); q["bad_datetime"] += int(bad.sum()); d = d[~bad]
    oy = d.rt.dt.year != YEAR; q["out_of_year"] += int(oy.sum()); d = d[~oy]
    dur = (d.rr - d.rt).dt.total_seconds() / 60
    short = (d.station == d.ret_station) & (dur < 2); q["short_same_station"] += int(short.sum()); d = d[~short]
    unk = d.station.isna(); q["unknown_station"] += int(unk.sum()); d = d[~unk]
    parts.append(d)
d = pd.concat(parts, ignore_index=True)
d["date"] = d.rt.dt.strftime("%Y-%m-%d")
q["used"] = len(d)
q["round_trip"] = int((d.station == d.ret_station).sum())
q["months_covered"] = sorted(d.rt.dt.strftime("%Y-%m").unique().tolist())
q["station_coord_match_rate"] = round(float(d.station.isin(set(st.station_id)).mean()), 4)

d.groupby(["station", "date"]).size().rename("n").reset_index() \
    .to_csv(os.path.join(OUT, "station_daily_rentals.csv"), index=False, encoding="utf-8-sig")
d.dropna(subset=["ret_station"]).assign(date=d.rr.dt.strftime("%Y-%m-%d")) \
    .groupby(["ret_station", "date"]).size().rename("n").reset_index().rename(columns={"ret_station": "station"}) \
    .to_csv(os.path.join(OUT, "station_daily_returns.csv"), index=False, encoding="utf-8-sig")
d.assign(hour=d.rt.dt.hour).groupby(["date", "hour"]).size().rename("n").reset_index() \
    .to_csv(os.path.join(OUT, "hourly.csv"), index=False, encoding="utf-8-sig")
json.dump(q, open(os.path.join(OUT, "quality.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 3) 대여소 요약 (eda_bike.station_table 과 같은 정의)
sd = pd.read_csv(os.path.join(OUT, "station_daily_rentals.csv"), dtype={"station": str})
g = sd.groupby("station").agg(total=("n", "sum"), first=("date", "min"), last=("date", "max"))
g["active_days"] = (pd.to_datetime(g["last"]) - pd.to_datetime(g["first"])).dt.days + 1
g["daily_mean"] = g.total / g.active_days
rd = pd.read_csv(os.path.join(OUT, "station_daily_returns.csv"), dtype={"station": str})
g["returns"] = rd.groupby("station")["n"].sum(); g["returns"] = g["returns"].fillna(0)
g["net_ratio"] = (g.returns - g.total) / (g.returns + g.total)
g = g.join(st.set_index("station_id")[["lat", "lon"]])
act = g[g.active_days >= MIN_ACTIVE_DAYS].reset_index()
act[["station", "daily_mean", "total", "active_days", "lat", "lon", "net_ratio"]] \
    .to_csv(os.path.join(OUT, "station_summary.csv"), index=False, encoding="utf-8-sig")

print("\n=== 세종 요약 ===")
print("사용 건수", f"{q['used']:,}", "/ 원본", f"{q['raw']:,}")
print("대여소 전체", len(g), "/ 60일+", len(act), "/ 좌표 있음", int(act.lat.notna().sum()))
print("대여소 일평균 중앙값", round(act.daily_mean.median(), 2))
print("좌표 매칭률(건수)", q["station_coord_match_rate"])
print("저장:", OUT)
