# -*- coding: utf-8 -*-
"""
구 경계가 보이는 실제 지도 (서울·대전·창원)

    python viz_map.py

입력: data/processed/features_station.csv, data/raw/boundary/skorea-municipalities-2018-geo.json
출력: outputs/figures/maps/*.png, outputs/gu_summary.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from features import to_utmk

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "figures" / "maps"
GEO = next((ROOT / "data" / "raw" / "boundary").glob("*municipalities*.json"))
CITIES = {"seoul": "서울 따릉이", "daejeon": "대전 타슈", "changwon": "창원 누비자"}
CELL = 250        # 격자 크기 (m)
SIGMA = 350       # 부드럽게 퍼뜨리는 정도 (m)
HEAT = LinearSegmentedColormap.from_list("heat", ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"])
HALO = [patheffects.withStroke(linewidth=3, foreground="white")]


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if n in have:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams.update({"axes.unicode_minus": False, "savefig.dpi": 220})


def load_gu():
    d = json.load(open(GEO, encoding="utf-8"))
    sel = {"seoul": lambda p: str(p["code"]).startswith("11"),
           "daejeon": lambda p: str(p["code"]).startswith("25"),
           "changwon": lambda p: str(p["name"]).startswith("창원시")}
    gus = {c: [] for c in CITIES}
    for f in d["features"]:
        p, g = f["properties"], f["geometry"]
        for c, ok in sel.items():
            if ok(p):
                polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
                rings = []
                for poly in polys:
                    for ring in poly:
                        a = np.asarray(ring)
                        x, y = to_utmk(a[:, 1], a[:, 0])
                        rings.append(np.c_[x, y])
                name = p["name"].replace("창원시", "")
                gus[c].append({"name": name, "rings": rings, "path": make_path(rings)})
    return gus


def make_path(rings):
    verts, codes = [], []
    for r in rings:
        verts += r.tolist() + [r[0].tolist()]
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 1) + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


def inside(path, x, y):
    return path.contains_points(np.c_[x, y])


def assign_gu(df, gus):
    df = df.copy()
    df["gu"] = None
    for c, lst in gus.items():
        m = df.city == c
        for g in lst:
            hit = m & df.gu.isna() & inside(g["path"], df.x, df.y)
            df.loc[hit, "gu"] = g["name"]
    return df


def extent(lst, pad=800):
    allp = np.vstack([r for g in lst for r in g["rings"]])
    return allp[:, 0].min() - pad, allp[:, 0].max() + pad, allp[:, 1].min() - pad, allp[:, 1].max() + pad


def density(g, ext):
    """대여소 이용량을 250m 격자에 가우시안으로 퍼뜨린 '이용 밀도' (건/일/km²)"""
    x0, x1, y0, y1 = ext
    xs, ys = np.arange(x0, x1, CELL) + CELL / 2, np.arange(y0, y1, CELL) + CELL / 2
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    sx, sy, w = g.x.values, g.y.values, g.daily_mean.values
    for i in range(0, len(sx), 200):
        dx = X[..., None] - sx[None, None, i:i + 200]
        dy = Y[..., None] - sy[None, None, i:i + 200]
        Z += (np.exp(-(dx ** 2 + dy ** 2) / (2 * SIGMA ** 2)) * w[i:i + 200]).sum(-1)
    Z = Z / (2 * np.pi * SIGMA ** 2) * 1e6
    return xs, ys, X, Y, Z


def draw_gu(ax, lst, lw=1.6, label=True, fs=9):
    for g in lst:
        ax.add_patch(PathPatch(g["path"], facecolor="none", edgecolor="white", lw=lw + 0.6, zorder=3))
        ax.add_patch(PathPatch(g["path"], facecolor="none", edgecolor="#7a7a7a", lw=0.6, zorder=3.1))
        if label:
            big = max(g["rings"], key=len)
            ax.text(big[:, 0].mean(), big[:, 1].mean(), g["name"], ha="center", va="center", fontsize=fs,
                    color="#333333", path_effects=HALO, zorder=5)


def outline(ax, lst):
    for g in lst:
        ax.add_patch(PathPatch(g["path"], facecolor="none", edgecolor="#9a9a9a", lw=0.6, zorder=2))


def frame(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.axis("off")
    x0, y0 = ext[0] + 1200, ext[2] + 1200
    ax.plot([x0, x0 + 5000], [y0, y0], color="black", lw=3, solid_capstyle="butt", zorder=6)
    ax.text(x0 + 2500, y0 + 500, "5 km", ha="center", fontsize=9, zorder=6)


def map_heat(ax, c, df, lst, vmax):
    g = df[df.city == c]
    ext = extent(lst)
    xs, ys, X, Y, Z = density(g, ext)
    city_path = make_path([r for gg in lst for r in gg["rings"]])
    mask = np.zeros(X.shape, bool)
    for gg in lst:
        mask |= inside(gg["path"], X.ravel(), Y.ravel()).reshape(X.shape)
    Zm = np.where(mask, Z, np.nan)
    im = ax.imshow(Zm, origin="lower", extent=(xs[0] - CELL / 2, xs[-1] + CELL / 2, ys[0] - CELL / 2, ys[-1] + CELL / 2),
                   cmap=HEAT, norm=LogNorm(vmin=10, vmax=vmax), interpolation="bilinear", zorder=1)
    outline(ax, lst)
    draw_gu(ax, lst)
    ax.scatter(g.x, g.y, s=2.5, color="#1f3a93", alpha=0.55, linewidths=0, zorder=4)
    frame(ax, ext)
    return im, ext


def map_choro(ax, c, gs, lst, norm, cmap):
    for gg in lst:
        row = gs[(gs.city == c) & (gs.gu == gg["name"])]
        v = row["대여소당_일평균_중앙값"].iloc[0] if len(row) else np.nan
        col = cmap(norm(v)) if not np.isnan(v) else "#eeeeee"
        ax.add_patch(PathPatch(gg["path"], facecolor=col, edgecolor="white", lw=1.6, zorder=2))
        big = max(gg["rings"], key=len)
        txt = f"{gg['name']}\n{v:.0f}건" if not np.isnan(v) else gg["name"]
        ax.text(big[:, 0].mean(), big[:, 1].mean(), txt, ha="center", va="center", fontsize=8.5,
                color="#222222", path_effects=HALO, zorder=5)
    frame(ax, extent(lst))


def main():
    font()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "features_station.csv", dtype={"station": str})
    gus = load_gu()
    df = assign_gu(df, gus)

    # 구별 요약
    gs = df.dropna(subset=["gu"]).groupby(["city", "gu"]).agg(
        대여소수=("station", "count"), 하루_총대여=("daily_mean", "sum"), 대여소당_일평균_중앙값=("daily_mean", "median"),
        상가_500m_중앙값=("shop_500", "median"), 지하철역거리_중앙값_m=("dist_subway_m", "median")).reset_index()
    gs["city_name"] = gs.city.map(CITIES)
    gs.sort_values(["city", "대여소당_일평균_중앙값"], ascending=[True, False]).to_csv(
        ROOT / "outputs" / "gu_summary.csv", index=False, encoding="utf-8-sig")
    print(gs.to_string(index=False))

    # 1) 도시별 이용 밀도 지도 (개별 파일, 발표용 큰 크기)
    vmax = 3000
    for c in CITIES:
        fig, ax = plt.subplots(figsize=(9, 8))
        im, _ = map_heat(ax, c, df, gus[c], vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("대여 밀도 (건/일/km²)")
        n = (df.city == c).sum()
        ax.set_title(f"{CITIES[c]} — 대여소 이용 밀도와 구 경계  (점: 대여소 {n:,}개)", loc="left", fontsize=13)
        fig.savefig(OUT / f"heat_{c}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("저장:", OUT / f"heat_{c}.png")

    # 2) 세 도시 한 장에 (같은 색 척도)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.25, 0.9, 1.1]})
    for ax, c in zip(axes, CITIES):
        im, _ = map_heat(ax, c, df, gus[c], vmax)
        ax.set_title(CITIES[c], loc="left", fontsize=13)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.012, pad=0.01)
    cb.set_label("대여 밀도 (건/일/km²)")
    fig.suptitle("세 도시 대여 밀도 — 같은 색 척도 (점: 대여소, 흰 선: 구 경계)", x=0.01, ha="left", fontsize=15)
    fig.savefig(OUT / "heat_three_cities.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", OUT / "heat_three_cities.png")

    # 3) 구별 대여소당 이용 (단계구분도)
    cmap = plt.get_cmap("Reds")
    for c in CITIES:
        sub = gs[gs.city == c]
        norm = plt.Normalize(vmin=0, vmax=max(10, sub["대여소당_일평균_중앙값"].max()))
        fig, ax = plt.subplots(figsize=(9, 8))
        map_choro(ax, c, gs, gus[c], norm, cmap)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("대여소 1곳당 하루 대여 (구별 중앙값)")
        ax.set_title(f"{CITIES[c]} — 구별 대여소당 이용", loc="left", fontsize=13)
        fig.savefig(OUT / f"gu_{c}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("저장:", OUT / f"gu_{c}.png")


if __name__ == "__main__":
    main()
