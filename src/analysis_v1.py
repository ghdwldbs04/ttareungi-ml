"""
따릉이 성공 요인의 타 지역 적용 가능성 — Phase 1 (EDA) + Phase 2 (H1~H4 통계 검증)

Phase 1  EDA
  1-1 도시별 이용량 분포 (표 + 그림)
  1-2 도시별 입지 변수 분포 (표 + 그림)
  1-3 공간 자기상관 Moran's I (도시별, kNN 가중치)
  1-4 도시별 이용량 지도
Phase 2  통계 검증
  H1/H2  변수별 Spearman 상관 (도시별, BH 보정) + 서울과 부호 일치 여부
  H4     도시별 변동성(최빈값 비율)·서울 범위 겹침 진단
  H4     변수×도시 상호작용 회귀 (HC3 강건 표준오차) → 요인×도시 전이 분류

모델 기반 검증(공간 CV, 기준선, 보정, LOCO, 그룹 중요도, SHAP)은 train_v2.py에서 수행.

실행 : 프로젝트 루트에서  python src/analysis_v1.py
결과 : results/analysis_v1/ 아래 csv·png
필요 패키지 : pandas numpy scipy scikit-learn matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "analysis_v1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CITIES = ["seoul", "daejeon", "sejong", "changwon"]
CITY_KO = {"seoul": "서울", "daejeon": "대전", "sejong": "세종", "changwon": "창원"}
DROP = ["city", "lat", "lon", "x", "y"]

GROUPS = {
    "인구": ["grid_pop_total", "grid_households", "grid_housing"],
    "고용·사업체": ["grid_businesses", "grid_workers"],
    "상업": ["shop_300", "shop_500", "shop_1000", "food_500"],
    "버스": ["bus_300", "bus_500", "bus_1000"],
    "지하철": ["dist_subway_m", "subway_500", "subway_1000"],
    "공원": ["dist_park_m", "park_500", "park_area_500"],
    "대학": ["dist_univ_m", "univ_300", "univ_500", "univ_1000"],
    "경사": ["slope_value"],
}
REP = {  # 상호작용 회귀 대표 변수 (docs/hypotheses.md 2절)
    "인구": "grid_pop_total", "고용·사업체": "grid_workers", "상업": "shop_500",
    "버스": "bus_500", "지하철": "subway_1000", "공원": "park_500",
    "대학": "dist_univ_m", "경사": "slope_value",
}
FEAT2GROUP = {f: g for g, fs in GROUPS.items() for f in fs}

# 판단 기준 (docs/hypotheses.md H4)
MODE_SHARE_MAX = 0.95   # 최빈값 비율 ≥ 95% → 식별 불가
OVERLAP_MIN = 0.50      # 서울 5~95% 범위 안 비율 < 50% → 범위 밖
ALPHA = 0.05


def set_font():
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                                   "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load():
    df = pd.read_csv(DATA / "stations_features_v2.csv", encoding="utf-8-sig")
    cols = pd.read_csv(DATA / "model_feature_columns_v2.csv", encoding="utf-8-sig")["column"].tolist()
    feats = [c for c in cols if c not in DROP]
    return df, feats


def bh(p):
    """Benjamini-Hochberg 보정 p값"""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(ranked, 0, 1)
    out[ok] = tmp
    return out


# ================================================================ Phase 1
def eda_target(df):
    t = df.groupby("city")["daily_mean"].describe(percentiles=[.1, .25, .5, .75, .9])
    t["cv"] = t["std"] / t["mean"]
    t = t.loc[CITIES]
    t.to_csv(OUT / "1_1_target_by_city.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [df.loc[df.city == c, "daily_mean"] for c in CITIES]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticks(range(1, 5), [CITY_KO[c] for c in CITIES])
    ax[0].set_title("도시별 일평균 대여 건수 (원래 단위, 이상치 표시 생략)")
    ax[0].set_ylabel("건/일")
    for c in CITIES:
        ax[1].hist(df.loc[df.city == c, "log_y"], bins=40, alpha=.45, density=True, label=CITY_KO[c])
    ax[1].set_title("도시별 이용량 분포 (로그 변환 단위)")
    ax[1].set_xlabel("log1p(건/일)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "1_1_target_by_city.png", dpi=150)
    plt.close(fig)
    return t


def eda_features(df, feats):
    rows = []
    for f in feats:
        r = {"feature": f, "group": FEAT2GROUP.get(f, "결측 표시")}
        for c in CITIES:
            s = df.loc[df.city == c, f]
            r[f"{c}_median"] = s.median()
            r[f"{c}_p90"] = s.quantile(.9)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "1_2_feature_by_city.csv", index=False, encoding="utf-8-sig")

    reps = list(REP.values())
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), reps):
        data = [np.log1p(df.loc[df.city == c, f].dropna().clip(lower=0)) for c in CITIES]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, 5), [CITY_KO[c] for c in CITIES])
        ax.set_title(f"{FEAT2GROUP[f]}: {f}\n(log1p)")
    fig.suptitle("요인 그룹 대표 변수의 도시별 분포")
    fig.tight_layout()
    fig.savefig(OUT / "1_2_feature_by_city.png", dpi=150)
    plt.close(fig)
    return t


def morans_i(xy, z, k=8, n_perm=199, rng=None):
    """kNN(행 표준화) 가중치 Moran's I + 순열 검정 p값"""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    zc = z - z.mean()
    denom = (zc ** 2).sum()
    I = np.sum(zc * zc[idx].mean(axis=1)) / denom
    rng = rng or np.random.default_rng(SEED)
    perms = np.array([np.sum((p := rng.permutation(zc)) * p[idx].mean(axis=1)) / denom
                      for _ in range(n_perm)])
    p = (np.sum(perms >= I) + 1) / (n_perm + 1)
    return I, p, -1 / (len(z) - 1)


def eda_spatial(df):
    rows = []
    rng = np.random.default_rng(SEED)
    for c in CITIES:
        d = df[df.city == c]
        I, p, EI = morans_i(d[["x", "y"]].values, d["log_y"].values, rng=rng)
        rows.append({"city": c, "n": len(d), "moran_I": I, "expected_I": EI, "p_perm": p})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "1_3_morans_i.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    vmin, vmax = df["log_y"].quantile([.02, .98])
    for ax, c in zip(axes, CITIES):
        d = df[df.city == c].sort_values("log_y")
        sc = ax.scatter(d["lon"], d["lat"], c=d["log_y"], s=6, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{CITY_KO[c]} (n={len(d)})")
        ax.set_aspect(1 / np.cos(np.deg2rad(d["lat"].mean())))
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, shrink=.8, label="log1p(일평균 대여 건수)")
    fig.suptitle("도시별 대여소 이용량 지도 (공통 색 척도)")
    fig.savefig(OUT / "1_4_usage_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return t


# ================================================================ Phase 2
def spearman_by_city(df, feats):
    rows = []
    for c in CITIES:
        d = df[df.city == c]
        for f in feats:
            s = d[[f, "log_y"]].dropna()
            if s[f].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(s[f], s["log_y"])
            rows.append({"city": c, "feature": f, "group": FEAT2GROUP.get(f, "결측 표시"),
                         "rho": rho, "p": p, "n": len(s)})
    t = pd.DataFrame(rows)
    t["p_bh"] = t.groupby("city")["p"].transform(bh)
    t["sig"] = t["p_bh"] < ALPHA

    w = t.pivot(index="feature", columns="city", values="rho")[CITIES]
    seoul_sign = np.sign(w["seoul"])
    for c in CITIES[1:]:
        w[f"{c}_same_sign"] = np.where(w[c].isna(), np.nan, (np.sign(w[c]) == seoul_sign).astype(float))
    w.insert(0, "group", [FEAT2GROUP.get(f, "결측 표시") for f in w.index])
    w = w.reindex(w["seoul"].abs().sort_values(ascending=False).index)

    t.to_csv(OUT / "2_1_spearman_long.csv", index=False, encoding="utf-8-sig")
    w.to_csv(OUT / "2_1_spearman_wide.csv", encoding="utf-8-sig")

    agree = {c: np.nanmean(w[f"{c}_same_sign"]) for c in CITIES[1:]}
    pd.Series(agree, name="sign_agreement_with_seoul").to_csv(
        OUT / "2_1_sign_agreement.csv", encoding="utf-8-sig")

    base = [f for f in w.index if not f.endswith("_missing")]
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(w.loc[base, CITIES].values, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([CITY_KO[c] for c in CITIES])
    ax.set_yticks(range(len(base))); ax.set_yticklabels([f"{f} ({FEAT2GROUP[f]})" for f in base], fontsize=8)
    for i, f in enumerate(base):
        for j, c in enumerate(CITIES):
            v = w.loc[f, c]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.6, label="Spearman ρ (vs log_y)")
    ax.set_title("변수별 이용량 상관 — 도시 비교\n('—' = 도시 내 값이 일정해 계산 불가)")
    fig.tight_layout()
    fig.savefig(OUT / "2_1_spearman_heatmap.png", dpi=150)
    plt.close(fig)
    return w, agree


def variability(df, feats):
    """도시별 변동성(최빈값 비율)과 서울 5~95% 범위 겹침 비율"""
    seoul = df[df.city == "seoul"]
    rows = []
    for f in feats:
        if f.endswith("_missing"):
            continue
        lo, hi = seoul[f].quantile([.05, .95])
        for c in CITIES:
            s = df.loc[df.city == c, f].dropna()
            mode_share = s.value_counts(normalize=True).iloc[0]
            overlap = ((s >= lo) & (s <= hi)).mean()
            status = ("식별 불가" if mode_share >= MODE_SHARE_MAX
                      else "범위 밖" if overlap < OVERLAP_MIN else "비교 가능")
            rows.append({"feature": f, "group": FEAT2GROUP[f], "city": c, "n_unique": s.nunique(),
                         "mode_value": s.mode().iloc[0], "mode_share": mode_share,
                         "std": s.std(), "seoul_p05": lo, "seoul_p95": hi,
                         "overlap_seoul_5_95": overlap, "status": status})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "2_2_variability_overlap.csv", index=False, encoding="utf-8-sig")
    st = t.pivot(index="feature", columns="city", values="status")[CITIES]
    st.to_csv(OUT / "2_2_status_matrix.csv", encoding="utf-8-sig")
    return t


# ---------------------------------------------------------------- 상호작용 회귀
def ols_hc3(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / (1 - h)) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return beta, cov, r2


def wald(beta, cov, idx):
    b, V = beta[idx], cov[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, stats.chi2.sf(stat, len(idx))


def interaction_regression(df, var_tbl):
    """log_y ~ 도시 더미 + Σ z_k + Σ z_k × 도시 더미 + 결측 통제
       z_k = log1p(대표 변수)를 서울 평균·표준편차로 표준화 → 기울기 단위를 도시 간 동일하게"""
    d = df.copy()
    for c in ["grid_pop_total", "grid_workers", "slope_value"]:
        d[c] = d.groupby("city")[c].transform(lambda s: s.fillna(s.median()))
    seoul = d.city == "seoul"

    cols, names = [np.ones(len(d))], ["const"]
    for c in CITIES[1:]:
        cols.append((d.city == c).astype(float).values); names.append(f"city[{c}]")
    cols.append(d["grid_any_missing"].values.astype(float)); names.append("grid_any_missing")
    cols.append(d["slope_missing"].values.astype(float)); names.append("slope_missing")

    status = var_tbl.set_index(["feature", "city"])["status"]
    z = {}
    for g, f in REP.items():
        v = np.log1p(d[f].clip(lower=0))
        z[g] = ((v - v[seoul].mean()) / v[seoul].std()).values
        cols.append(z[g]); names.append(f"{g}")
    inter_idx = {g: {} for g in REP}
    for g, f in REP.items():
        for c in CITIES[1:]:
            if status.get((f, c)) == "식별 불가":
                continue  # 도시 내 변동이 없으면 상호작용 추정 불가 → 제외
            cols.append(z[g] * (d.city == c).values); names.append(f"{g}×{c}")
            inter_idx[g][c] = len(names) - 1

    X = np.column_stack(cols)
    y = d["log_y"].values
    beta, cov, r2 = ols_hc3(X, y)
    se = np.sqrt(np.diag(cov))
    coef = pd.DataFrame({"term": names, "coef": beta, "se_hc3": se,
                         "z": beta / se, "p": 2 * stats.norm.sf(np.abs(beta / se))})
    coef.to_csv(OUT / "2_3_interaction_coefs.csv", index=False, encoding="utf-8-sig")

    rows, joint = [], []
    for g, f in REP.items():
        bi = names.index(g)
        rows.append({"group": g, "rep_var": f, "city": "seoul", "slope": beta[bi], "se": se[bi],
                     "p_slope": 2 * stats.norm.sf(abs(beta[bi] / se[bi])),
                     "p_interaction": np.nan, "status": "기준(서울)"})
        for c in CITIES[1:]:
            st = status.get((f, c))
            if c not in inter_idx[g]:
                rows.append({"group": g, "rep_var": f, "city": c, "slope": np.nan, "se": np.nan,
                             "p_slope": np.nan, "p_interaction": np.nan, "status": "식별 불가"})
                continue
            ii = inter_idx[g][c]
            L = np.zeros(len(beta)); L[bi] = 1; L[ii] = 1
            slope, s_se = L @ beta, np.sqrt(L @ cov @ L)
            p_int = coef.loc[ii, "p"]
            same = np.sign(slope) == np.sign(beta[bi])
            if st == "범위 밖":
                label = "범위 밖"
            elif p_int >= ALPHA:
                label = "전이됨" if same else "전이됨(부호 다르나 차이 비유의)"
            else:
                label = "크기 차이" if same else "방향 불일치"
            rows.append({"group": g, "rep_var": f, "city": c, "slope": slope, "se": s_se,
                         "p_slope": 2 * stats.norm.sf(abs(slope / s_se)),
                         "p_interaction": p_int, "status": label})
        if inter_idx[g]:
            W, pj = wald(beta, cov, list(inter_idx[g].values()))
            joint.append({"group": g, "rep_var": f, "n_interactions": len(inter_idx[g]),
                          "wald_chi2": W, "p_joint": pj})
    slopes = pd.DataFrame(rows)
    slopes.to_csv(OUT / "2_3_city_slopes_and_status.csv", index=False, encoding="utf-8-sig")
    joint = pd.DataFrame(joint)
    joint.to_csv(OUT / "2_3_joint_wald.csv", index=False, encoding="utf-8-sig")

    mat = slopes.pivot(index="group", columns="city", values="status")[CITIES].loc[list(REP)]
    mat.to_csv(OUT / "2_3_transfer_matrix.csv", encoding="utf-8-sig")

    # 그림: 도시별 기울기 (95% CI)
    fig, ax = plt.subplots(figsize=(10, 6))
    off = {c: (i - 1.5) * 0.18 for i, c in enumerate(CITIES)}
    for c in CITIES:
        s = slopes[slopes.city == c].set_index("group").loc[list(REP)]
        yy = np.arange(len(REP)) + off[c]
        ax.errorbar(s["slope"], yy, xerr=1.96 * s["se"], fmt="o", ms=4, capsize=2, label=CITY_KO[c])
    ax.axvline(0, color="gray", lw=.8)
    ax.set_yticks(range(len(REP))); ax.set_yticklabels([f"{g}\n({f})" for g, f in REP.items()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("표준화 기울기 (서울 기준 1SD 증가당 log_y 변화, 95% CI)")
    ax.set_title(f"요인별 도시 기울기 — 상호작용 회귀 (R²={r2:.3f})\n점이 없으면 식별 불가")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "2_3_city_slopes.png", dpi=150)
    plt.close(fig)
    return mat, joint, r2


# ================================================================ main
def main():
    set_font()
    df, feats = load()
    print(f"데이터 {df.shape}, 입력 피처 {len(feats)}개\n")
    pd.set_option("display.width", 180)

    print("=== Phase 1-1 도시별 이용량 (건/일) ===")
    print(eda_target(df)[["count", "mean", "50%", "std", "cv"]].round(2).to_string())

    eda_features(df, feats)
    print("\n=== Phase 1-3 Moran's I (log_y, kNN=8) ===")
    print(eda_spatial(df).round(4).to_string(index=False))

    print("\n=== H1/H2 Spearman 상관 (서울 |ρ| 상위 12개) ===")
    w, agree = spearman_by_city(df, feats)
    print(w[["group"] + CITIES].head(12).round(3).to_string())
    print("서울과 부호 일치율:", {CITY_KO[k]: round(v, 3) for k, v in agree.items()})

    print("\n=== H4 변동성·범위 진단 (대표 변수) ===")
    var_tbl = variability(df, feats)
    print(var_tbl[var_tbl.feature.isin(REP.values())]
          .pivot(index="feature", columns="city", values="status")[CITIES].to_string())

    print("\n=== H4 상호작용 회귀: 요인 × 도시 전이 분류 ===")
    mat, joint, r2 = interaction_regression(df, var_tbl)
    print(f"R² = {r2:.3f}")
    print(mat.to_string())
    print("\n요인별 도시 간 효과 차이 결합 검정 (Wald):")
    print(joint.round(4).to_string(index=False))

    print(f"\n저장 완료 → {OUT}")


if __name__ == "__main__":
    main()
