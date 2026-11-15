# -*- coding: utf-8 -*-
"""
지역별·공간별 현황 시각화 (발표용 PNG)

    python viz_spatial.py

입력: data/processed/features_station.csv, data/processed/feature_sources/sgis.csv
출력: outputs/figures/spatial/*.png
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "figures" / "spatial"
CITIES = {"seoul": "서울 따릉이", "daejeon": "대전 타슈", "changwon": "창원 누비자"}
COLORS = {"seoul": "#2a6fdb", "daejeon": "#e07b39", "changwon": "#3a9e6b"}
WIN_KM = 40  # 세 도시 지도를 같은 축척(40km × 40km)으로

LABELS = {
    "shop_500": "상가 수 (500m)", "food_500": "음식점 수 (500m)", "firms_1km": "사업체 수 (1km 격자)",
    "workers_1km": "종사자 수 (1km 격자)", "pop_1km": "인구 (1km 격자)", "young_share_1km": "20~39세 비율",
    "stations_500": "주변 대여소 수 (500m)", "dist_nn_station_m": "가장 가까운 대여소 거리",
    "dist_subway_m": "지하철역 거리", "subway_1000": "지하철역 수 (1km)", "bus_500": "버스정류장 수 (500m)",
    "dist_univ_m": "대학 거리", "dist_park_m": "공원 거리", "park_area_500": "공원 면적 (500m)",
}


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if n in have:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams.update({"axes.unicode_minus": False, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", OUT / name)


def sgis_cells():
    p = PROC / "feature_sources" / "sgis.csv"
    if not p.exists():
        return None
    g = pd.read_csv(p)
    L = "가나다라마바사아자차카"
    code = g["격자코드"].astype(str)
    g["x0"] = code.str[0].map(L.index).add(7) * 100_000 + code.str[2:4].astype(int) * 1000
    g["y0"] = code.str[1].map(L.index).add(13) * 100_000 + code.str[4:6].astype(int) * 1000
    return g


def window(g):
    cx, cy = g.x.median(), g.y.median()
    h = WIN_KM * 500
    return cx - h, cx + h, cy - h, cy + h


def underlay(ax, cells, col, win, cmap="Greys", vmax=None):
    x0, x1, y0, y1 = win
    c = cells[(cells.x0 >= x0 - 1000) & (cells.x0 <= x1) & (cells.y0 >= y0 - 1000) & (cells.y0 <= y1)]
    if len(c):
        nx, ny = int((x1 - x0) // 1000) + 2, int((y1 - y0) // 1000) + 2
        M = np.full((ny, nx), np.nan)
        ix = ((c.x0 - x0) // 1000).astype(int).clip(0, nx - 1)
        iy = ((c.y0 - y0) // 1000).astype(int).clip(0, ny - 1)
        M[iy, ix] = c[col].values
        M[M <= 0] = np.nan
        ax.imshow(M, origin="lower", extent=(x0, x0 + nx * 1000, y0, y0 + ny * 1000), cmap=cmap,
                  norm=LogNorm(vmin=100, vmax=vmax or np.nanmax(M)), alpha=0.55, interpolation="nearest", zorder=0)


def scalebar(ax, win, km=5):
    x0, x1, y0, y1 = win
    bx, by = x0 + 1500, y0 + 1500
    ax.plot([bx, bx + km * 1000], [by, by], color="black", lw=3, solid_capstyle="butt")
    ax.text(bx + km * 500, by + 600, f"{km} km", ha="center", fontsize=9)


def fig_maps(df, cells):
    """1) 세 도시 같은 축척 지도: 인구(회색 바탕) 위 대여소별 일평균 대여"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9))
    norm = LogNorm(vmin=0.5, vmax=200)
    for ax, c in zip(axes, CITIES):
        g = df[df.city == c].sort_values("daily_mean")
        win = window(g)
        if cells is not None:
            underlay(ax, cells, "pop", win, vmax=60_000)
        sc = ax.scatter(g.x, g.y, c=g.daily_mean.clip(0.5), cmap="plasma", norm=norm, s=9, linewidths=0, zorder=2)
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#cccccc")
        ax.set_title(f"{CITIES[c]}  ·  대여소 {len(g):,}개  ·  중앙값 {g.daily_mean.median():.1f}건/일", loc="left", fontsize=12)
        scalebar(ax, win)
    cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cb.set_label("대여소 일평균 대여건수 (log)")
    fig.suptitle("같은 축척(40km)으로 본 세 도시 — 서울은 넓은 지역을 촘촘하게, 대전·창원은 도심에 집중  (회색: 1km 격자 인구)",
                 x=0.01, ha="left", fontsize=14)
    save(fig, "01_three_city_maps_same_scale.png")


def fig_hotspots(df):
    """2) 도시 안 상대 순위(상위 20% / 하위 20%)로 핫스팟·콜드스팟"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9))
    for ax, c in zip(axes, CITIES):
        g = df[df.city == c]
        win = window(g)
        mid = g[(g.rank_pct > 0.2) & (g.rank_pct <= 0.8)]
        top, low = g[g.rank_pct > 0.8], g[g.rank_pct <= 0.2]
        ax.scatter(mid.x, mid.y, s=6, color="#d9d9d9", linewidths=0, label="중간 60%")
        ax.scatter(low.x, low.y, s=10, color="#4a90d9", linewidths=0, label="하위 20%")
        ax.scatter(top.x, top.y, s=12, color="#d7301f", linewidths=0, label="상위 20%")
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#cccccc")
        ax.set_title(f"{CITIES[c]} — 상위 20% 기준 {top.daily_mean.min():.0f}건/일 이상", loc="left", fontsize=12)
        scalebar(ax, win)
    axes[0].legend(loc="upper left", frameon=False, markerscale=2)
    fig.suptitle("도시 안에서 '잘 되는 대여소'는 어디에 몰려 있나 (도시별 상대 순위)", x=0.01, ha="left", fontsize=14)
    save(fig, "02_hotspots_relative_rank.png")


def fig_city_summary(df):
    """3) 도시 규모 vs 이용 강도"""
    rows = []
    for c in CITIES:
        g = df[df.city == c]
        rows.append({"city": c, "대여소 수": len(g), "하루 총 대여": g.daily_mean.sum(),
                     "대여소당 하루 대여(중앙값)": g.daily_mean.median(),
                     "가장 가까운 대여소 거리(중앙값, m)": g.dist_nn_station_m.median()})
    s = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, col in zip(axes, s.columns[1:]):
        b = ax.bar([CITIES[c].split()[0] for c in s.city], s[col], color=[COLORS[c] for c in s.city])
        ax.set_title(col, loc="left", fontsize=11)
        ax.bar_label(b, labels=[f"{v:,.0f}" if v >= 100 else f"{v:,.1f}" for v in s[col]], padding=3, fontsize=10)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
    fig.suptitle("규모가 아니라 '대여소 1곳당 이용 강도'에서 도시 차이가 드러난다", x=0.01, ha="left", fontsize=14, y=1.05)
    save(fig, "03_city_scale_vs_intensity.png")


def fig_corr_heatmap(df, feats):
    """4) 도시별로 각 요인과 이용량의 순위상관 — 요인이 도시마다 같은 방향인가"""
    M = pd.DataFrame({CITIES[c].split()[0]: [df[df.city == c][[f, "log_y"]].corr(method="spearman").iloc[0, 1] for f in feats]
                      for c in CITIES}, index=[LABELS.get(f, f) for f in feats])
    M = M.sort_values("서울", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(M) + 1.2))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(3), M.columns); ax.set_yticks(range(len(M)), M.index)
    ax.xaxis.tick_top()
    for i in range(len(M)):
        for j in range(3):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.35 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, label="이용량과의 순위상관 (Spearman)")
    ax.set_title("요인별 이용량과의 관계 — 도시마다 같은 방향인가", loc="left", fontsize=12, pad=28)
    save(fig, "04_factor_correlation_by_city.png")
    M.to_csv(OUT / "04_factor_correlation_by_city.csv", encoding="utf-8-sig")


def fig_binned(df, feats):
    """5) 핵심 요인 구간별(서울 기준 10분위) 이용량 — 관계의 모양이 도시마다 같은가"""
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, f in zip(axes.flat, feats):
        edges = np.unique(df.loc[df.city == "seoul", f].quantile(np.linspace(0, 1, 11)).values)
        for c in CITIES:
            g = df[df.city == c]
            b = pd.cut(g[f], edges, include_lowest=True)
            m = g.groupby(b, observed=False)["daily_mean"].agg(["median", "size"])
            mids = [(iv.left + iv.right) / 2 for iv in m.index]
            ok = m["size"] >= 10
            ax.plot(np.array(mids)[ok], m["median"][ok], marker="o", lw=2, color=COLORS[c], label=CITIES[c])
        ax.set_title(LABELS.get(f, f), loc="left", fontsize=11)
        ax.set_ylabel("대여소 일평균 대여 (구간 중앙값)")
        if f in ("shop_500", "food_500", "workers_1km", "firms_1km", "pop_1km", "park_area_500"):
            ax.set_xscale("symlog")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("같은 조건이면 같은 성과가 나는가 — 요인 구간별 이용량 (구간은 서울 10분위, 대여소 10개 미만 구간 생략)",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, "05_same_condition_same_result.png")


def fig_coverage(df):
    """6) 서울 범위 밖에 있는 대전·창원 대여소 비율"""
    p = ROOT / "outputs" / "feature_ood.csv"
    if not p.exists():
        return
    o = pd.read_csv(p)
    o = o[o.feature.isin(LABELS.keys())]
    o["label"] = o.feature.map(lambda f: LABELS.get(f, f))
    o = o.sort_values("changwon_out_of_seoul_range_%")
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(o) + 1))
    y = np.arange(len(o))
    ax.barh(y + 0.2, o["daejeon_out_of_seoul_range_%"], height=0.4, color=COLORS["daejeon"], label="대전")
    ax.barh(y - 0.2, o["changwon_out_of_seoul_range_%"], height=0.4, color=COLORS["changwon"], label="창원")
    ax.set_yticks(y, o.label)
    ax.set_xlabel("서울 대여소 범위(1~99%)를 벗어난 대여소 비율 (%)")
    ax.axvline(10, color="gray", ls=":", lw=1)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("서울에서 배운 모델이 '본 적 없는 환경' — 지하철이 가장 큰 차이", loc="left", fontsize=12)
    save(fig, "06_out_of_seoul_range.png")


def main():
    font()
    df = pd.read_csv(PROC / "features_station.csv", dtype={"station": str})
    feats = [f for f in LABELS if f in df.columns]
    fig_city_summary(df)
    fig_corr_heatmap(df, feats)
    fig_binned(df, ["shop_500", "workers_1km", "pop_1km", "stations_500", "dist_subway_m", "bus_500"])
    fig_coverage(df)


if __name__ == "__main__":
    main()
