# -*- coding: utf-8 -*-
"""
따릉이(서울)·타슈(대전)·누비자(창원) 대여이력 EDA

사용법 (ttareungi 폴더에서):
    python eda_bike.py inspect   # 1) 파일·컬럼 자동 인식 결과 확인 -> outputs/00_schema_report.txt
    python eda_bike.py build     # 2) 대용량 이력을 chunk로 읽어 집계 -> data/processed/*.csv
    python eda_bike.py plots     # 3) 그림(PNG) + 요약표 -> outputs/figures, outputs/eda_summary.md
    python eda_bike.py all       # 1~3 한 번에

폴더 구조 (CSV 또는 zip 그대로 넣어도 됨):
    data/raw/seoul/rentals/     서울 대여이력 (zip/csv)
    data/raw/seoul/stations/    서울 대여소 정보 (xlsx/csv)
    data/raw/daejeon/rentals/   타슈 대여이력 (csv/zip)
    data/raw/daejeon/stations/  타슈 위치 현황 (선택, 이력에 GPS가 있으면 없어도 됨)
    data/raw/changwon/rentals/  누비자 대여 반납 이력
    data/raw/changwon/stations/ 누비자 터미널 정보
"""
import io
import json
import math
import re
import sys
import time
import zipfile
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
# 설정 (필요하면 여기만 바꾸기)
# ============================================================
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

YEAR = 2025                       # 기본 분석 연도
# 도시별 분석 기간 (대전 타슈는 공개 이력이 2025-12 ~ 2026-10 이라 따로 지정)
PERIOD = {
    "seoul": ("2025-01-01", "2025-12-31"),
    "changwon": ("2025-01-01", "2025-12-31"),
    "daejeon": ("2025-12-01", "2026-10-31"),
}
CHUNK = 500_000                 # 한 번에 읽을 행 수 (메모리 부족하면 500_000)
GRID_M = 250                      # 격자 크기 (m)
MIN_ACTIVE_DAYS = 60              # 운영일수 이 미만인 대여소는 대여소 분석에서 제외
CITIES = {"seoul": "서울 따릉이", "daejeon": "대전 타슈", "changwon": "창원 누비자"}
CITY_COLORS = {"seoul": "#2a6fdb", "daejeon": "#e07b39", "changwon": "#3a9e6b"}

# 자동 인식이 틀리면 여기에 직접 컬럼명을 적기 (inspect 결과를 보고)
# 예: OVERRIDES = {"daejeon": {"rent_station": "대여소ID", "rent_dt": "대여일시"}}
OVERRIDES = {
    # 타슈 이력의 X좌표가 위도(36.x), Y좌표가 경도(127.x)
    "daejeon": {"rent_lat": "대여_X좌표", "rent_lon": "대여_Y좌표"},
}

# ============================================================
# 공통 유틸
# ============================================================
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def set_korean_font():
    cands = {"Darwin": ["AppleGothic", "Apple SD Gothic Neo"],
             "Windows": ["Malgun Gothic"],
             "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]}
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in cands.get(platform.system(), []) + ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def sniff_encoding(raw_bytes):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            raw_bytes.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "cp949"


class Source:
    """CSV 파일 하나 또는 zip 안의 CSV 하나"""

    def __init__(self, path, member=None):
        self.path, self.member = Path(path), member

    @property
    def name(self):
        return f"{self.path.name}::{self.member}" if self.member else self.path.name

    def open_bytes(self):
        if self.member:
            return zipfile.ZipFile(self.path).open(self.member)
        return open(self.path, "rb")

    def head_bytes(self, n=200_000):
        with self.open_bytes() as f:
            return f.read(n)

    def encoding(self):
        b = self.head_bytes()
        # 잘린 멀티바이트 문자 때문에 실패하지 않도록 마지막 줄 제거
        b = b[: b.rfind(b"\n")] if b"\n" in b else b
        return sniff_encoding(b)

    def read(self, **kw):
        enc = self.encoding()
        f = self.open_bytes()
        return pd.read_csv(io.TextIOWrapper(f, encoding=enc, errors="replace"), low_memory=False, **kw)


def list_sources(folder, exts=(".csv", ".txt")):
    srcs = []
    if not folder.exists():
        return srcs
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts:
            srcs.append(Source(p))
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for m in z.namelist():
                    if Path(m).suffix.lower() in exts and not m.startswith("__MACOSX"):
                        srcs.append(Source(p, m))
    return srcs


# ============================================================
# 컬럼 자동 인식
# ============================================================
RENT_WORDS = ("대여", "출발", "rent", "start")
RET_WORDS = ("반납", "도착", "return", "end")


def _side(c, side):
    """컬럼이 대여(출발) 쪽인지 반납(도착) 쪽인지: 먼저 나오는 단어로 판단
    예) '반납대여소번호' -> 반납, '대여 대여소번호' -> 대여"""
    c = norm(c)
    pos = lambda ws: min([c.find(w) for w in ws if w in c] or [10 ** 6])
    r, t = pos(RENT_WORDS), pos(RET_WORDS)
    if r == t == 10 ** 6:
        return False
    return (r < t) if side == "rent" else (t < r)


def detect_rental_columns(cols, city):
    cols = list(cols)
    n = {c: norm(c) for c in cols}
    found = {}

    def pick(cands):
        for c in cols:
            if cands(n[c], c):
                return c
        return None

    for side in ("rent", "ret"):
        s = "rent" if side == "rent" else "ret"
        side_name = "rent" if side == "rent" else "return"
        # 일시 (한 컬럼)
        found[f"{side_name}_dt"] = pick(lambda x, c: _side(c, s) and ("일시" in x or "datetime" in x or x.endswith("시각")))
        if not found[f"{side_name}_dt"]:
            d = pick(lambda x, c: _side(c, s) and ("일자" in x or x.endswith("일") or "date" in x))
            t = pick(lambda x, c: _side(c, s) and (("시간" in x or "시각" in x or "time" in x) and "이용" not in x))
            if d and t and d != t:
                found[f"{side_name}_date"], found[f"{side_name}_time"] = d, t
        # 대여소 ID / 번호
        found[f"{side_name}_station"] = (
            pick(lambda x, c: _side(c, s) and any(k in x for k in ("대여소", "터미널", "정류", "station", "스테이션")) and any(k in x for k in ("번호", "id", "코드", "code")))
            or pick(lambda x, c: _side(c, s) and any(k in x for k in ("터미널", "정류장", "대여소", "station")) and "명" not in x and "주소" not in x)
            or pick(lambda x, c: _side(c, s) and any(k in x for k in ("대여소명", "터미널명", "정류장명", "name")))
        )
        found[f"{side_name}_name"] = pick(lambda x, c: _side(c, s) and ("명" in x or "name" in x) and "주소" not in x)
        found[f"{side_name}_lat"] = pick(lambda x, c: _side(c, s) and ("위도" in x or "lat" in x))
        found[f"{side_name}_lon"] = pick(lambda x, c: _side(c, s) and ("경도" in x or "lon" in x or "lng" in x))
    found["duration"] = pick(lambda x, c: "이용시간" in x or "duration" in x or "사용시간" in x)
    found["distance"] = pick(lambda x, c: "이용거리" in x or "distance" in x or "이동거리" in x)
    found["bike_id"] = pick(lambda x, c: "자전거번호" in x or "bike" in x)
    found.update(OVERRIDES.get(city, {}))
    return {k: v for k, v in found.items() if v}


def detect_station_table(path):
    """대여소/터미널 정보 파일: 헤더 행을 찾아 id/name/lat/lon/capacity 반환"""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = Source(path).read(header=None, dtype=str)
    hdr = None
    for i in range(min(15, len(raw))):
        row = " ".join(norm(v) for v in raw.iloc[i].tolist())
        if ("위도" in row or "lat" in row) and ("경도" in row or "lon" in row or "lng" in row):
            hdr = i
            break
    if hdr is None:
        return None, f"{path.name}: 위도/경도 헤더를 찾지 못함"
    # 여러 줄로 된 머리글(병합 셀) 처리: 위도 칸에 숫자가 처음 나오는 행부터 데이터
    lat_j = next(j for j, v in enumerate(raw.iloc[hdr].tolist()) if "위도" in norm(v) or "lat" in norm(v))
    start = hdr + 1
    while start < len(raw) and pd.isna(pd.to_numeric(raw.iat[start, lat_j], errors="coerce")):
        start += 1
    cols = []
    for j in range(raw.shape[1]):
        parts = [re.sub(r"\s+", " ", str(v)).strip() for v in raw.iloc[:start, j].tolist() if str(v) != "nan"]
        cols.append(" ".join(dict.fromkeys(parts)) or f"col{j}")
    df = raw.iloc[start:].copy()
    df.columns = cols
    n = {c: norm(c) for c in cols}

    def pick(f):
        for c in cols:
            if f(n[c]):
                return c
        return None

    sid = pick(lambda x: any(k in x for k in ("대여소번호", "대여소id", "터미널번호", "터미널id", "정류장코드", "정류소코드", "터미널코드", "stationid", "station_id"))) \
        or pick(lambda x: x in ("번호", "id", "코드", "code"))
    name = pick(lambda x: any(k in x for k in ("대여소명", "터미널명", "정류장명", "정류소명", "보관소", "name"))) or pick(lambda x: x.endswith("명"))
    lat = pick(lambda x: "위도" in x or x.startswith("lat"))
    lon = pick(lambda x: "경도" in x or x.startswith("lon") or x.startswith("lng"))
    capc = [c for c in cols if any(k in n[c] for k in ("거치대", "용량", "수용", "capacity"))]
    df = df.loc[:, ~pd.Index(cols).duplicated()]
    out = pd.DataFrame({
        "station_id": df[sid] if sid else np.nan,
        "station_name": df[name] if name else np.nan,
        "lat": pd.to_numeric(df[lat], errors="coerce"),
        "lon": pd.to_numeric(df[lon], errors="coerce"),
    })
    if capc:
        out["capacity"] = df[capc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    out = out.dropna(subset=["lat", "lon"])
    out = out[(out.lat.between(33, 39)) & (out.lon.between(124, 132))]
    return out, f"{path.name}: header rows 0-{start - 1}, id={sid}, name={name}, lat={lat}, lon={lon}, capacity={capc}"


def norm_id(s):
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(~s.str.fullmatch(r"\d+"), s.str.lstrip("0").replace("", "0"))


# ============================================================
# 1) inspect
# ============================================================
def cmd_inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for city in CITIES:
        lines.append(f"\n==================== {city} ====================")
        srcs = list_sources(RAW / city / "rentals")
        lines.append(f"[이력 파일] {len(srcs)}개")
        if not srcs:
            lines.append(f"  -> {RAW / city / 'rentals'} 폴더에 파일이 없음")
        for s in srcs[:3]:
            try:
                df = s.read(nrows=5)
                lines.append(f"\n  {s.name}  (encoding={s.encoding()})")
                lines.append(f"  columns: {list(df.columns)}")
                lines.append(f"  detected: {json.dumps(detect_rental_columns(df.columns, city), ensure_ascii=False)}")
                lines.append("  sample:\n" + "\n".join("    " + l for l in df.head(3).to_string().splitlines()))
            except Exception as e:
                lines.append(f"  {s.name}: 읽기 실패 {e!r}")
        if len(srcs) > 3:
            lines.append(f"  ... 외 {len(srcs) - 3}개 (첫 파일과 컬럼이 같다고 가정, build 단계에서 파일마다 다시 인식)")
        st_dir = RAW / city / "stations"
        st_files = [p for p in st_dir.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")] if st_dir.exists() else []
        lines.append(f"\n[대여소 파일] {len(st_files)}개")
        for p in st_files:
            try:
                st, msg = detect_station_table(p)
                lines.append(f"  {msg}" + (f"  -> {len(st)}개 대여소" if st is not None else ""))
            except Exception as e:
                lines.append(f"  {p.name}: 읽기 실패 {e!r}")
    text = "\n".join(lines)
    (OUT / "00_schema_report.txt").write_text(text, encoding="utf-8")
    print(text)
    log(f"저장: {OUT / '00_schema_report.txt'}")


# ============================================================
# 2) build : chunk 단위 집계
# ============================================================
class Acc:
    """groupby 합계를 여러 chunk에 걸쳐 누적 (메모리 절약용으로 주기적으로 압축)"""

    def __init__(self, keys):
        self.keys, self.parts = keys, []

    def add(self, df):
        if len(df):
            self.parts.append(df)
        if len(self.parts) >= 8:
            self.compact()

    def compact(self):
        if self.parts:
            self.parts = [pd.concat(self.parts).groupby(self.keys, as_index=False, observed=True).sum()]

    def result(self):
        self.compact()
        return self.parts[0] if self.parts else pd.DataFrame(columns=self.keys)


def parse_dt(df, cols, side):
    if f"{side}_dt" in cols:
        return pd.to_datetime(df[cols[f"{side}_dt"]].astype(str).str.strip(), errors="coerce", cache=True)
    if f"{side}_date" in cols:
        d = df[cols[f"{side}_date"]].astype(str).str.strip()
        t = df[cols[f"{side}_time"]].astype(str).str.strip()
        t = t.where(~t.str.fullmatch(r"\d{3,6}"), t.str.zfill(6).str.replace(r"(\d{2})(\d{2})(\d{2})", r"\1:\2:\3", regex=True))
        return pd.to_datetime(d + " " + t, errors="coerce", cache=True)
    return pd.Series(pd.NaT, index=df.index)


def build_city(city):
    srcs = list_sources(RAW / city / "rentals")
    if not srcs:
        log(f"{city}: 이력 파일 없음, 건너뜀")
        return None
    st_day = Acc(["station", "date"])
    ret_day = Acc(["station", "date"])
    hourly = Acc(["date", "hour"])
    od = Acc(["station", "ret_station"])
    coords = Acc(["station"])
    dur_bins = np.arange(0, 182, 2)
    dist_bins = np.arange(0, 20_250, 250)
    dur_hist = np.zeros(len(dur_bins) - 1)
    dist_hist = np.zeros(len(dist_bins) - 1)
    q = dict(rows_total=0, bad_datetime=0, out_of_year=0, short_same_station=0, round_trip=0, used=0)
    months = set()

    for s in srcs:
        enc = s.encoding()
        head = s.read(nrows=5)
        cols = detect_rental_columns(head.columns, city)
        need = [v for k, v in cols.items() if not k.endswith("_name") and k != "bike_id"]
        if "rent_station" not in cols or not any(k in cols for k in ("rent_dt", "rent_date")):
            log(f"  !! {s.name}: 대여소/대여일시 컬럼을 못 찾음 -> 건너뜀. inspect 결과를 보고 OVERRIDES에 지정")
            continue
        log(f"{city} | {s.name} | enc={enc}")
        with s.open_bytes() as f:
            reader = pd.read_csv(io.TextIOWrapper(f, encoding=enc, errors="replace"), usecols=list(dict.fromkeys(need)),
                                 dtype=str, chunksize=CHUNK, low_memory=False)
            for ch in reader:
                q["rows_total"] += len(ch)
                rt = parse_dt(ch, cols, "rent")
                q["bad_datetime"] += int(rt.isna().sum())
                p0, p1 = PERIOD.get(city, (f"{YEAR}-01-01", f"{YEAR}-12-31"))
                inside = (rt >= pd.Timestamp(p0)) & (rt < pd.Timestamp(p1) + pd.Timedelta(days=1))
                ok = rt.notna() & inside
                q["out_of_year"] += int((rt.notna() & ~inside).sum())
                ch, rt = ch[ok], rt[ok]
                if ch.empty:
                    continue
                d = pd.DataFrame({"station": norm_id(ch[cols["rent_station"]]), "date": rt.dt.date.astype(str),
                                  "hour": rt.dt.hour.astype("int16")})
                d["ret_station"] = norm_id(ch[cols["return_station"]]) if "return_station" in cols else np.nan
                # 이용시간(분)
                if "duration" in cols:
                    dur = pd.to_numeric(ch[cols["duration"]], errors="coerce")
                else:
                    rr = parse_dt(ch, cols, "return")
                    dur = (rr - rt).dt.total_seconds() / 60
                dur = dur.where(dur.between(0, 24 * 60))
                short = (dur < 2) & (d["station"] == d["ret_station"])
                q["short_same_station"] += int(short.sum())
                d, dur = d[~short.values], dur[~short.values]
                q["round_trip"] += int((d["station"] == d["ret_station"]).sum())
                q["used"] += len(d)
                months.update(pd.to_datetime(d["date"]).dt.to_period("M").astype(str).unique())
                dur_hist += np.histogram(dur.dropna().clip(upper=dur_bins[-1] - 1e-6), bins=dur_bins)[0]
                if "distance" in cols:
                    dist = pd.to_numeric(ch.loc[~short.values, cols["distance"]], errors="coerce")
                    if "km" in norm(cols["distance"]):
                        dist = dist * 1000
                    dist_hist += np.histogram(dist.dropna().clip(0, dist_bins[-1] - 1e-6), bins=dist_bins)[0]
                d["n"] = 1
                st_day.add(d.groupby(["station", "date"], as_index=False)["n"].sum())
                hourly.add(d.groupby(["date", "hour"], as_index=False)["n"].sum())
                if d["ret_station"].notna().any():
                    r = d.dropna(subset=["ret_station"])
                    od.add(r.groupby(["station", "ret_station"], as_index=False)["n"].sum())
                    if "return_dt" in cols or "return_date" in cols:
                        rr = parse_dt(ch.loc[~short.values], cols, "return")
                        rd = pd.DataFrame({"station": d["ret_station"].values, "date": rr.dt.date.astype(str).values, "n": 1})
                        ret_day.add(rd[rr.notna().values].groupby(["station", "date"], as_index=False)["n"].sum())
                if "rent_lat" in cols and "rent_lon" in cols:
                    c = pd.DataFrame({"station": d["station"].values,
                                      "lat": pd.to_numeric(ch.loc[~short.values, cols["rent_lat"]], errors="coerce").values,
                                      "lon": pd.to_numeric(ch.loc[~short.values, cols["rent_lon"]], errors="coerce").values})
                    c = c.dropna()
                    c = c[c.lat.between(33, 39) & c.lon.between(124, 132)]
                    c["k"] = 1
                    coords.add(c.groupby("station", as_index=False)[["lat", "lon", "k"]].sum())
                log(f"    누적 {q['used']:,}건 사용 / {q['rows_total']:,}건 읽음")

    out = PROC / city
    out.mkdir(parents=True, exist_ok=True)
    sd = st_day.result()
    sd.to_csv(out / "station_daily_rentals.csv", index=False, encoding="utf-8-sig")
    rd = ret_day.result()
    if len(rd):
        rd.to_csv(out / "station_daily_returns.csv", index=False, encoding="utf-8-sig")
    hourly.result().to_csv(out / "hourly.csv", index=False, encoding="utf-8-sig")
    o = od.result()
    if len(o):
        o.sort_values("n", ascending=False).to_csv(out / "od_pairs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"min_from": dur_bins[:-1], "count": dur_hist}).to_csv(out / "duration_hist.csv", index=False)
    if dist_hist.sum() > 0:
        pd.DataFrame({"m_from": dist_bins[:-1], "count": dist_hist}).to_csv(out / "distance_hist.csv", index=False)

    # 대여소 좌표: (1) 대여소 정보 파일, (2) 이력 안의 GPS 평균
    st = None
    st_dir = RAW / city / "stations"
    for p in sorted(st_dir.rglob("*")) if st_dir.exists() else []:
        if p.suffix.lower() in (".xlsx", ".xls", ".csv"):
            t, msg = detect_station_table(p)
            log("  대여소 파일:", msg)
            if t is not None and len(t):
                t["station_id"] = norm_id(t["station_id"]) if t["station_id"].notna().any() else t["station_id"]
                st = t if st is None else pd.concat([st, t])
    c = coords.result()
    if len(c):
        c = c.assign(lat=c.lat / c.k, lon=c.lon / c.k).drop(columns="k").rename(columns={"station": "station_id"})
        c["station_name"] = np.nan
        st = c if st is None else pd.concat([st, c[~c.station_id.isin(st.station_id)]])
    if st is not None:
        st = st.drop_duplicates("station_id", keep="last")
        hit = sd["station"].isin(st["station_id"])
        if hit.mean() < 0.5 and st["station_name"].notna().any():
            log("  ID 매칭률이 낮아 대여소 이름으로 다시 매칭")
            st2 = st.assign(station_id=st["station_name"].astype(str).str.strip())
            if sd["station"].isin(st2["station_id"]).mean() > hit.mean():
                st = st2
        st.to_csv(out / "stations.csv", index=False, encoding="utf-8-sig")
        q["station_coord_match_rate"] = round(float(sd.groupby("station")["n"].sum().index.isin(st["station_id"]).mean()), 3)
    q["months_covered"] = sorted(months)
    q["n_files"] = len(srcs)
    (out / "quality.json").write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{city} 완료: {json.dumps(q, ensure_ascii=False)}")
    return q


def cmd_build():
    for city in CITIES:
        build_city(city)


# ============================================================
# 3) plots
# ============================================================
def load(city, name):
    p = PROC / city / name
    return pd.read_csv(p, dtype={"station": str, "station_id": str, "ret_station": str}) if p.exists() else None


def station_table(city):
    sd = load(city, "station_daily_rentals.csv")
    if sd is None or sd.empty:
        return None
    g = sd.groupby("station").agg(total=("n", "sum"), first=("date", "min"), last=("date", "max"), days_with_trip=("date", "nunique"))
    g["active_days"] = (pd.to_datetime(g["last"]) - pd.to_datetime(g["first"])).dt.days + 1
    g["daily_mean"] = g["total"] / g["active_days"]
    rd = load(city, "station_daily_returns.csv")
    if rd is not None:
        g["returns"] = rd.groupby("station")["n"].sum()
        g["returns"] = g["returns"].fillna(0)
        g["net_ratio"] = (g["returns"] - g["total"]) / (g["returns"] + g["total"])
    st = load(city, "stations.csv")
    if st is not None:
        g = g.join(st.drop_duplicates("station_id").set_index("station_id")[[c for c in ("station_name", "lat", "lon", "capacity") if c in st.columns]])
    g["city"] = city
    return g.reset_index()


def to_xy(lat, lon, lat0):
    """간이 평면좌표(m). 도시 하나 안에서 250m 격자 만들기엔 충분한 정확도"""
    return lon * 111_320 * math.cos(math.radians(lat0)), lat * 110_540


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("  그림 저장:", name)


def cmd_plots():
    set_korean_font()
    FIG.mkdir(parents=True, exist_ok=True)
    cities = [c for c in CITIES if (PROC / c / "station_daily_rentals.csv").exists()]
    if not cities:
        log("집계 파일이 없음. 먼저 python eda_bike.py build")
        return
    stab = {c: station_table(c) for c in cities}
    hourly = {c: load(c, "hourly.csv") for c in cities}
    summary = []

    # 01 일별 이용 추이 (도시별 패널, 7일 이동평균)
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 2.6 * len(cities)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cities):
        dd = hourly[c].groupby("date")["n"].sum()
        dd.index = pd.to_datetime(dd.index)
        dd = dd.asfreq("D")
        ax.plot(dd.index, dd.values, color=CITY_COLORS[c], alpha=0.25, lw=0.8)
        ax.plot(dd.index, dd.rolling(7, min_periods=3).mean(), color=CITY_COLORS[c], lw=2)
        ax.set_title(f"{CITIES[c]} — 일별 대여건수 (굵은 선: 7일 이동평균)", loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        missing = dd[dd.isna()].index
        if len(missing):
            ax.text(0.99, 0.9, f"데이터 없는 날 {len(missing)}일", transform=ax.transAxes, ha="right", fontsize=9, color="crimson")
    savefig(fig, "01_daily_trend.png")

    # 02 요일×시간 히트맵 (도시 내 비중 %, 세 도시 같은 색 척도)
    wd = ["월", "화", "수", "목", "금", "토", "일"]
    mats = {}
    for c in cities:
        h = hourly[c].copy()
        h["wd"] = pd.to_datetime(h["date"]).dt.dayofweek
        m = h.pivot_table(index="wd", columns="hour", values="n", aggfunc="sum").reindex(index=range(7), columns=range(24)).fillna(0)
        mats[c] = m / m.values.sum() * 100
    vmax = max(float(m.values.max()) for m in mats.values())
    fig, axes = plt.subplots(1, len(cities), figsize=(4.8 * len(cities), 3.6), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cities):
        im = ax.imshow(mats[c].values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_yticks(range(7), wd)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("시")
        ax.set_title(CITIES[c], fontsize=11, loc="left")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, label="전체 대여 중 비중(%)")
    fig.suptitle("요일·시간대별 이용 비중 — 출퇴근형인지 여가형인지", x=0.01, ha="left", fontsize=12)
    savefig(fig, "02_weekday_hour_heatmap.png")

    # 03 시간대 프로파일: 평일 vs 주말 (도시 비교)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for c in cities:
        h = hourly[c].copy()
        h["we"] = pd.to_datetime(h["date"]).dt.dayofweek >= 5
        for ax, we in zip(axes, (False, True)):
            p = h[h.we == we].groupby("hour")["n"].sum().reindex(range(24), fill_value=0)
            ax.plot(p.index, p / p.sum() * 100, color=CITY_COLORS[c], lw=2, label=CITIES[c])
    for ax, t in zip(axes, ("평일", "주말·토일")):
        ax.set_title(t, loc="left")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("시")
    axes[0].set_ylabel("하루 대여 중 비중(%)")
    axes[0].legend(frameon=False)
    savefig(fig, "03_hourly_profile_weekday_weekend.png")

    # 04 대여소별 일평균 대여 분포 + 집중도(Lorenz)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in cities:
        g = stab[c]
        g = g[g.active_days >= MIN_ACTIVE_DAYS]
        v = g["daily_mean"].clip(lower=0.01)
        axes[0].hist(np.log10(v), bins=40, histtype="step", lw=2, color=CITY_COLORS[c], density=True, label=f"{CITIES[c]} (n={len(g):,})")
        s = np.sort(g["total"].values)[::-1]
        cum = np.cumsum(s) / s.sum()
        x = np.arange(1, len(s) + 1) / len(s) * 100
        axes[1].plot(x, cum * 100, color=CITY_COLORS[c], lw=2, label=CITIES[c])
        top10 = cum[max(int(len(s) * 0.1) - 1, 0)] * 100
        summary.append((c, "상위 10% 대여소의 대여 비중(%)", round(top10, 1)))
    axes[0].set_xlabel("대여소 일평균 대여건수 (log10)")
    axes[0].set_xticks([-1, 0, 1, 2], ["0.1", "1", "10", "100"])
    axes[0].set_title("대여소별 일평균 대여건수 분포", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot([0, 100], [0, 100], color="gray", ls=":", lw=1)
    axes[1].set_xlabel("이용량 상위 대여소 비율(%)")
    axes[1].set_ylabel("누적 대여 비중(%)")
    axes[1].set_title("이용 집중도 — 소수 대여소에 몰려 있는가", loc="left")
    savefig(fig, "04_station_distribution_concentration.png")

    # 05 이용시간·거리 분포
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for c in cities:
        dh = load(c, "duration_hist.csv")
        if dh is not None and dh["count"].sum() > 0:
            axes[0].plot(dh["min_from"] + 1, dh["count"] / dh["count"].sum() * 100, color=CITY_COLORS[c], lw=2, label=CITIES[c])
            cum = (dh["count"].cumsum() / dh["count"].sum())
            summary.append((c, "이용시간 중앙값(분, 2분 구간)", float(dh.loc[cum >= 0.5, "min_from"].iloc[0] + 1)))
        di = load(c, "distance_hist.csv")
        if di is not None and di["count"].sum() > 0:
            axes[1].plot((di["m_from"] + 125) / 1000, di["count"] / di["count"].sum() * 100, color=CITY_COLORS[c], lw=2, label=CITIES[c])
    axes[0].set_xlim(0, 120)
    axes[0].set_xlabel("이용시간(분)")
    axes[0].set_ylabel("비중(%)")
    axes[0].set_title("이용시간 분포", loc="left")
    axes[0].legend(frameon=False)
    axes[1].set_xlim(0, 15)
    axes[1].set_xlabel("이용거리(km) — 단위가 m가 아닌 도시는 확인 필요")
    axes[1].set_title("이용거리 분포", loc="left")
    savefig(fig, "05_duration_distance.png")

    # 06 대여소 지도 + 07 250m 격자 지도
    grid_rows = []
    for c in cities:
        g = stab[c]
        if "lat" not in g.columns or g["lat"].notna().sum() == 0:
            log(f"  {c}: 좌표 없음 -> 지도 생략 (stations 폴더 확인)")
            continue
        g = g.dropna(subset=["lat", "lon"])
        g = g[g.active_days >= MIN_ACTIVE_DAYS]
        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(g.lon, g.lat, c=np.log10(g.daily_mean.clip(lower=0.1)), s=8, cmap="viridis", alpha=0.85, linewidths=0, vmin=-1, vmax=2)
        ax.set_aspect(1 / math.cos(math.radians(g.lat.mean())))
        cb = fig.colorbar(sc, ax=ax, fraction=0.035)
        cb.set_ticks([-1, 0, 1, 2], labels=["0.1", "1", "10", "100"])
        cb.set_label("일평균 대여건수")
        ax.set_title(f"{CITIES[c]} 대여소별 일평균 대여 ({len(g):,}개소)", loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        savefig(fig, f"06_station_map_{c}.png")

        lat0 = g.lat.mean()
        x, y = to_xy(g.lat.values, g.lon.values, lat0)
        g = g.assign(gx=(x // GRID_M).astype(int), gy=(y // GRID_M).astype(int))
        gr = g.groupby(["gx", "gy"]).agg(n_stations=("station", "count"), daily_mean_sum=("daily_mean", "sum"),
                                         lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
        gr["city"] = c
        grid_rows.append(gr)
        gx0, gy0 = gr.gx.min(), gr.gy.min()
        M = np.full((gr.gy.max() - gy0 + 1, gr.gx.max() - gx0 + 1), np.nan)
        M[gr.gy - gy0, gr.gx - gx0] = np.log10(gr.daily_mean_sum.clip(lower=0.1))
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(M, origin="lower", cmap="magma_r", interpolation="nearest", vmin=-1, vmax=3)
        cb = fig.colorbar(im, ax=ax, fraction=0.035)
        cb.set_ticks([-1, 0, 1, 2, 3], labels=["0.1", "1", "10", "100", "1000"])
        cb.set_label("격자 내 일평균 대여 합")
        ax.set_title(f"{CITIES[c]} {GRID_M}m 격자별 이용량 (대여소 있는 격자 {len(gr):,}개)", loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        savefig(fig, f"07_grid{GRID_M}m_{c}.png")
    if grid_rows:
        pd.concat(grid_rows).to_csv(PROC / f"grid_{GRID_M}m_usage.csv", index=False, encoding="utf-8-sig")

    # 08 월별: 활성 대여소 1곳당 하루 대여 (도시 간 공정 비교)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for c in cities:
        sd = load(c, "station_daily_rentals.csv")
        sd["month"] = sd["date"].str[:7]
        m = sd.groupby("month").agg(n=("n", "sum"), st=("station", "nunique"), days=("date", "nunique"))
        ax.plot(m.index, m.n / m.st / m.days, marker="o", color=CITY_COLORS[c], lw=2, label=CITIES[c])
    ax.set_ylabel("대여소 1곳당 하루 대여")
    ax.set_title("월별 대여소당 이용 강도 — 도시 규모를 빼고 비교", loc="left")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45)
    savefig(fig, "08_monthly_intensity.png")

    # 09 대여/반납 불균형
    if any("net_ratio" in stab[c].columns for c in cities):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for c in cities:
            g = stab[c]
            if "net_ratio" in g.columns:
                g = g[(g.active_days >= MIN_ACTIVE_DAYS) & (g.total >= 100)]
                ax.hist(g.net_ratio, bins=np.linspace(-1, 1, 41), histtype="step", lw=2, density=True, color=CITY_COLORS[c], label=CITIES[c])
                summary.append((c, "반납-대여 불균형 |비율|>0.2 대여소 비중(%)", round(float((g.net_ratio.abs() > 0.2).mean() * 100), 1)))
        ax.set_xlabel("(반납 − 대여) / (반납 + 대여)  ← 대여가 많음 | 반납이 많음 →")
        ax.set_title("대여소별 대여·반납 불균형 (재배치 수요)", loc="left")
        ax.legend(frameon=False)
        savefig(fig, "09_imbalance.png")

    # 요약표
    rows = []
    for c in cities:
        g = stab[c]
        q = json.loads((PROC / c / "quality.json").read_text(encoding="utf-8"))
        act = g[g.active_days >= MIN_ACTIVE_DAYS]
        days = hourly[c]["date"].nunique()
        rows.append({
            "도시": CITIES[c],
            "기간": f"{q['months_covered'][0]}~{q['months_covered'][-1]} ({len(q['months_covered'])}개월)" if q["months_covered"] else "-",
            "사용 건수": f"{q['used']:,}",
            "하루 평균 대여": f"{q['used'] / max(days, 1):,.0f}",
            "대여소 수(전체/60일+)": f"{len(g):,} / {len(act):,}",
            "대여소 일평균 대여 중앙값": round(float(act.daily_mean.median()), 2) if len(act) else None,
            "좌표 매칭률": q.get("station_coord_match_rate", "-"),
            "제거: 날짜오류/연도밖/2분미만 제자리": f"{q['bad_datetime']:,} / {q['out_of_year']:,} / {q['short_same_station']:,}",
            "제자리 반납 비율(%)": round(q["round_trip"] / max(q["used"], 1) * 100, 1),
        })
        act[["station", "daily_mean", "total", "active_days"] + [k for k in ("lat", "lon", "capacity", "net_ratio") if k in act.columns]] \
            .to_csv(PROC / c / "station_summary.csv", index=False, encoding="utf-8-sig")
    tab = pd.DataFrame(rows)
    extra = pd.DataFrame(summary, columns=["city", "지표", "값"])
    extra["city"] = extra["city"].map(CITIES)
    try:
        tab_txt = tab.to_markdown(index=False)
    except ImportError:  # tabulate 미설치
        tab_txt = tab.to_string(index=False)
    md = ["# EDA 요약", "", f"분석 연도: {YEAR}, 대여소 분석 기준: 운영 {MIN_ACTIVE_DAYS}일 이상", "",
          tab_txt, "",
          "## 추가 지표", "", extra.pivot(index="지표", columns="city", values="값").to_string(), "",
          "## 그림", ""] + [f"- figures/{p.name}" for p in sorted(FIG.glob("*.png"))]
    (OUT / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    log(f"요약 저장: {OUT / 'eda_summary.md'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    for d in [RAW / c / sub for c in CITIES for sub in ("rentals", "stations")] + [PROC, FIG]:
        d.mkdir(parents=True, exist_ok=True)
    if cmd in ("inspect", "all"):
        cmd_inspect()
    if cmd in ("build", "all"):
        cmd_build()
    if cmd in ("plots", "all"):
        cmd_plots()
