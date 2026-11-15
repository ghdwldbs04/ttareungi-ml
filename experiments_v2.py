# -*- coding: utf-8 -*-
"""
보완 실험 — 피드백(실험 타당성) 반영

    python experiments_v2.py           # 전체 실행 (로컬 PC 기준 10~20분)
    python experiments_v2.py --quick   # 반복 수를 줄여 오류만 빠르게 점검

필요: model.py, viz_map.py 와 같은 폴더 / data/processed/features_station_v2.csv / 각 도시 station_daily_rentals.csv
출력: outputs/model_v2/*.csv, outputs/model_v2/summary.txt, outputs/figures/model_v2/*.png

P0   대여소 개수 흐름(원본 → 60일+ → 좌표 → 모델 입력)과 단계별 평균·중앙값
P1   공통 월(1~6, 8~10, 12월)만으로 목표변수를 다시 만들어 E0·E1·E2 핵심 수치 재계산
E1c  최종 설정 전이: 대전 22개 / 창원 20개(자전거도로 제외, 서울 모델도 20개로 다시 학습)
E2v2 현지 보정: 무작위 추출 vs 공간 블록(2km)+500m 버퍼, 20회 반복 → 평균 ± 표준편차
E3v2 요인 중요도: 세 도시 모두 공간 그룹 CV, 10회 반복 → 평균 비중·표준편차·Top1/Top3 유지율
E3s  SHAP 세 도시 비교 (LightGBM 내장 SHAP 사용, shap 패키지 불필요)
E4v2 구별 오차를 실제 건수 기준 배율로 다시 계산 (log1p 변환 영향 확인)
G    수요 등급 지도: 서울 모델 예측 5등급 vs 실제 5등급 (대전·창원)
H    하이퍼파라미터·재현 조건 기록
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from model import gbm, ridge, font, LABELS, BASE_FEATS, EXTRA, SHARES, CITIES, COLORS, SEED

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "model_v2"
FIG = ROOT / "outputs" / "figures" / "model_v2"
OLD_FIG = ROOT / "outputs" / "figures" / "model"
QUICK = "--quick" in sys.argv
R_E2 = 4 if QUICK else 20
R_E3 = 2 if QUICK else 10
R_P1 = 2 if QUICK else 10
BLOCK = 2000        # 공간 블록 크기 (m)
BUFFER = 500        # 학습 대여소에서 이 거리 안의 평가 대여소는 제외 (m)
COMMON_MONTHS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12]   # 세 도시 모두 자료가 있는 달(월 기준)
FULL = BASE_FEATS + EXTRA
SAFE = [f for f in FULL if f not in ("cycleway_m_500", "dist_cycleway_m")]
FEATS_OF = {"seoul": FULL, "daejeon": FULL, "changwon": SAFE}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def X_of(d, feats, log=False):
    X = d[feats].astype(float).copy()
    if log:
        lf = [f for f in feats if f not in SHARES]
        X[lf] = np.log1p(X[lf].clip(lower=0))
    return X.fillna(X.median())


def rmse(y, p):
    return float(mean_squared_error(y, p) ** 0.5)


def sp(y, p):
    return float(spearmanr(y, p).correlation)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say("  그림:", FIG / name)


def to_csv(d, name, index=False):
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / name, index=index, encoding="utf-8-sig")
    say("  표:", OUT / name)


# ------------------------------------------------------------
def load():
    df = pd.read_csv(PROC / "features_station_v2.csv", dtype={"station": str})
    from viz_map import load_gu, assign_gu
    gus = load_gu()
    df = assign_gu(df, gus)
    df["gu"] = df["gu"].fillna("기타")
    df["block"] = df.city + "_" + (df.x // BLOCK).astype(int).astype(str) + "_" + (df.y // BLOCK).astype(int).astype(str)
    return df, gus


def p0_counts(df):
    say("\n[P0] 대여소 개수 흐름과 평균·중앙값")
    rows = []
    for c in CITIES:
        daily = pd.read_csv(PROC / c / "station_daily_rentals.csv", dtype={"station": str})
        summ = pd.read_csv(PROC / c / "station_summary.csv", dtype={"station": str})
        model_in = df[df.city == c]
        stages = [("① 대여 이력에 나온 대여소", daily.groupby("station").n.sum().index.size, None),
                  ("② 운영 60일 이상", len(summ), summ.daily_mean),
                  ("③ ② 중 좌표 확인", int(summ.lat.notna().sum()), summ.loc[summ.lat.notna(), "daily_mean"]),
                  ("④ 피처 생성·모델 입력", len(model_in), model_in.daily_mean)]
        for name, n, v in stages:
            rows.append({"도시": CITIES[c], "단계": name, "대여소 수": n,
                         "하루 대여 평균": None if v is None else round(float(v.mean()), 1),
                         "하루 대여 중앙값": None if v is None else round(float(v.median()), 1)})
    t = pd.DataFrame(rows)
    to_csv(t, "P0_station_counts.csv")
    say(t.to_string(index=False))
    return t


def common_period_target(df):
    """공통 월만으로 대여소 하루 평균을 다시 계산 (기존 정의와 동일: 첫 대여일~마지막 대여일 사이의 달력일 기준)"""
    out = []
    for c in CITIES:
        d = pd.read_csv(PROC / c / "station_daily_rentals.csv", dtype={"station": str}, parse_dates=["date"])
        span = d.groupby("station").date.agg(["min", "max"])
        d = d[d.date.dt.month.isin(COMMON_MONTHS)]
        tot = d.groupby("station").n.sum()
        allday = pd.date_range(span["min"].min(), span["max"].max(), freq="D")
        ok = pd.Series(allday.month.isin(COMMON_MONTHS).astype(int), index=allday).cumsum()
        lo = ok.reindex(span["min"] - pd.Timedelta(days=1)).fillna(0).values
        hi = ok.reindex(span["max"]).values
        days = pd.Series(hi - lo, index=span.index)
        r = pd.DataFrame({"city": c, "station": span.index, "days_c": days.values,
                          "total_c": tot.reindex(span.index).fillna(0).values})
        out.append(r)
    r = pd.concat(out)
    r = r[r.days_c >= 60]
    r["daily_mean_c"] = r.total_c / r.days_c
    df = df.merge(r[["city", "station", "daily_mean_c"]], on=["city", "station"], how="left")
    df["log_y_c"] = np.log1p(df.daily_mean_c)
    return df


# ------------------------------------------------------------
def split_random(T, frac, rng):
    idx = rng.permutation(len(T))
    k = int(round(frac * len(T)))
    return idx[:k], idx[k:]


def split_block(T, frac, rng):
    blocks = T.block.unique().copy()
    rng.shuffle(blocks)
    size = T.groupby("block").size()
    chosen, cnt = [], 0
    for b in blocks:
        if cnt >= frac * len(T):
            break
        chosen.append(b)
        cnt += size[b]
    is_tr = T.block.isin(chosen).values
    tr, te0 = np.where(is_tr)[0], np.where(~is_tr)[0]
    dist, _ = cKDTree(T[["x", "y"]].values[tr]).query(T[["x", "y"]].values[te0])
    return tr, te0[dist > BUFFER]


def e2_one(S, T, c, feats, ycol, frac, split, rng, m_seoul):
    tr, te = (split_block if split == "block" else split_random)(T, frac, rng)
    y_te = T[ycol].values[te]
    base = m_seoul.predict(X_of(T, feats))
    r = {"n_train": len(tr), "n_test": len(te), "train_share": len(tr) / len(T)}
    r["zero_RMSE"], r["zero_SP"] = rmse(y_te, base[te]), sp(y_te, base[te])
    off = np.mean(T[ycol].values[tr] - base[tr])
    r["intercept_RMSE"], r["intercept_SP"] = rmse(y_te, base[te] + off), sp(y_te, base[te] + off)
    loc = gbm().fit(X_of(T.iloc[tr], feats), T[ycol].iloc[tr]).predict(X_of(T.iloc[te], feats))
    r["local_RMSE"], r["local_SP"] = rmse(y_te, loc), sp(y_te, loc)
    pool = pd.concat([S, T.iloc[tr]])
    Xp = X_of(pool, feats).assign(is_local=(pool.city == c).astype(float).values)
    w = np.where(pool.city == c, 5.0, 1.0)
    pp = gbm().fit(Xp, pool[ycol], sample_weight=w).predict(X_of(T.iloc[te], feats).assign(is_local=1.0))
    r["pool_RMSE"], r["pool_SP"] = rmse(y_te, pp), sp(y_te, pp)
    return r


METHODS = {"zero": "서울 모델 그대로", "intercept": "서울 모델 + 절편 보정", "local": "현지 데이터만", "pool": "서울+현지 함께"}


def e2v2(df):
    say(f"\n[E2v2] 현지 보정 — 무작위 vs 공간 블록({BLOCK}m)+버퍼({BUFFER}m), {R_E2}회 반복")
    S = df[df.city == "seoul"].reset_index(drop=True)
    rows = []
    for c in ("daejeon", "changwon"):
        feats = FEATS_OF[c]
        T = df[df.city == c].reset_index(drop=True)
        m_seoul = gbm().fit(X_of(S, feats), S.log_y)
        for split in ("random", "block"):
            for frac in (0.1, 0.25, 0.5):
                for rep in range(R_E2):
                    rng = np.random.default_rng(SEED + 1000 * rep + int(frac * 100))
                    r = e2_one(S, T, c, feats, "log_y", frac, split, rng, m_seoul)
                    rows.append({"도시": CITIES[c], "분할": split, "현지비율": frac, "rep": rep, **r})
            say(f"  {CITIES[c]} {split} 완료")
    raw = pd.DataFrame(rows)
    to_csv(raw, "E2v2_raw.csv")
    cols = [f"{m}_{k}" for m in METHODS for k in ("RMSE", "SP")] + ["n_train", "n_test", "train_share"]
    agg = raw.groupby(["도시", "분할", "현지비율"])[cols].agg(["mean", "std"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    to_csv(agg.reset_index(), "E2v2_summary.csv")
    raw["pool_vs_local"] = raw.local_RMSE - raw.pool_RMSE
    raw["pool_vs_zero_pct"] = (raw.zero_RMSE - raw.pool_RMSE) / raw.zero_RMSE * 100
    pair = raw.groupby(["도시", "분할", "현지비율"]).agg(
        서울현지_개선율_평균=("pool_vs_zero_pct", "mean"), 서울현지_개선율_표준편차=("pool_vs_zero_pct", "std"),
        서울현지가_현지만보다_나은_비율=("pool_vs_local", lambda s: (s > 0).mean()),
        RMSE차이_현지만_minus_서울현지=("pool_vs_local", "mean")).round(3)
    to_csv(pair.reset_index(), "E2v2_paired.csv")
    say(agg[[f"{m}_RMSE_mean" for m in METHODS] + [f"{m}_RMSE_std" for m in METHODS] + ["n_test_mean"]].to_string())
    say(pair.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    st = {"intercept": ("#7a7a7a", "--"), "local": ("#e07b39", "-"), "pool": ("#2a6fdb", "-")}
    for ax, c in zip(axes, ("대전 타슈", "창원 누비자")):
        a = agg.loc[(c, "block")]
        ax.axhline(a["zero_RMSE_mean"].mean(), color="black", lw=1.2, ls=":", label="서울 모델 그대로")
        for m, (col, ls) in st.items():
            ax.errorbar(a.index * 100, a[f"{m}_RMSE_mean"], yerr=a[f"{m}_RMSE_std"], marker="o", lw=2, color=col,
                        ls=ls, capsize=4, label=f"{METHODS[m]} (공간 블록)")
        b = agg.loc[(c, "random")]
        ax.plot(b.index * 100, b["pool_RMSE_mean"], color="#2a6fdb", lw=1, ls=":", marker=".", alpha=0.7,
                label="서울+현지 함께 (무작위 추출, 참고)")
        ax.set_title(c, loc="left")
        ax.set_xticks([10, 25, 50])
        ax.set_xlabel("학습에 쓴 현지 대여소 비율 (%)")
    axes[0].set_ylabel("예측 오차 RMSE (log, 낮을수록 좋음)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"공간적으로 떨어진 지역을 평가해도 현지 데이터 보정 효과가 유지되는가 (막대: ±1 표준편차, {R_E2}회)",
                 x=0.01, ha="left", fontsize=12)
    save(fig, "E2v2_spatial_learning_curve.png")
    return agg, pair


# ------------------------------------------------------------
def e1_final(df):
    say("\n[E1c] 최종 설정 전이 (대전 22개, 창원 20개 — 서울 모델도 같은 피처로 학습)")
    S = df[df.city == "seoul"]
    rows = []
    for c in ("daejeon", "changwon"):
        feats = FEATS_OF[c]
        T = df[df.city == c]
        m = gbm().fit(X_of(S, feats), S.log_y)
        p = m.predict(X_of(T, feats))
        y = T.log_y.values
        rows.append({"도시": CITIES[c], "피처 수": len(feats), "RMSE(log)": rmse(y, p), "R2": r2_score(y, p),
                     "Spearman": sp(y, p), "평균편향": float(np.mean(p - y)),
                     "R2(편향 보정 후)": r2_score(y, p - np.mean(p - y))})
    t = pd.DataFrame(rows).round(3)
    to_csv(t, "E1c_final_setting.csv")
    say(t.to_string(index=False))
    return t


def p1_period(df):
    say(f"\n[P1] 공통 월 {COMMON_MONTHS} 만으로 목표변수를 다시 만들었을 때")
    d = df.dropna(subset=["log_y_c"]).reset_index(drop=True)
    rows = []
    for c in CITIES:
        t = d[d.city == c]
        rows.append({"항목": "대여소 하루 대여 중앙값", "도시": CITIES[c], "전체 기간": round(t.daily_mean.median(), 1),
                     "공통 월": round(t.daily_mean_c.median(), 1)})
    S = d[d.city == "seoul"].reset_index(drop=True)
    from sklearn.model_selection import GroupKFold
    for ycol, lab in (("log_y", "전체 기간"), ("log_y_c", "공통 월")):
        p = np.zeros(len(S))
        for tr, te in GroupKFold(5).split(S, S[ycol], S.gu):
            p[te] = gbm().fit(X_of(S.iloc[tr], FULL), S[ycol].iloc[tr]).predict(X_of(S.iloc[te], FULL))
        rows.append({"항목": "E0 서울 자치구 CV R²", "도시": "서울 따릉이", lab: round(r2_score(S[ycol], p), 3)})
        rows.append({"항목": "E0 서울 자치구 CV Spearman", "도시": "서울 따릉이", lab: round(sp(S[ycol], p), 3)})
        for c in ("daejeon", "changwon"):
            feats = FEATS_OF[c]
            T = d[d.city == c].reset_index(drop=True)
            m = gbm().fit(X_of(S, feats), S[ycol])
            pz = m.predict(X_of(T, feats))
            y = T[ycol].values
            rows += [{"항목": "E1 R²", "도시": CITIES[c], lab: round(r2_score(y, pz), 3)},
                     {"항목": "E1 Spearman", "도시": CITIES[c], lab: round(sp(y, pz), 3)},
                     {"항목": "E1 평균편향", "도시": CITIES[c], lab: round(float(np.mean(pz - y)), 3)}]
            res = [e2_one(S, T, c, feats, ycol, 0.1, "block", np.random.default_rng(SEED + k), m) for k in range(R_P1)]
            rows.append({"항목": "E2 공간블록 10% 서울+현지 RMSE", "도시": CITIES[c], lab: round(np.mean([r["pool_RMSE"] for r in res]), 3)})
            rows.append({"항목": "E2 공간블록 10% 서울모델 그대로 RMSE", "도시": CITIES[c], lab: round(np.mean([r["zero_RMSE"] for r in res]), 3)})
    t = pd.DataFrame(rows).groupby(["항목", "도시"], sort=False).first().reset_index()
    to_csv(t, "P1_common_period.csv")
    say(t.to_string(index=False))
    return t


# ------------------------------------------------------------
def e3v2(df):
    say(f"\n[E3v2] 도시별 요인 중요도 — 공간 그룹 CV, {R_E3}회 반복")
    shares, top, cvr2 = {}, {}, {}
    for c in CITIES:
        feats = FEATS_OF[c]
        T = df[df.city == c].reset_index(drop=True)
        grp = T.gu if c == "seoul" else T.block
        groups = grp.unique()
        reps, r2s = [], []
        for rep in range(R_E3):
            rng = np.random.default_rng(SEED + rep)
            fold_of = dict(zip(rng.permutation(groups), np.arange(len(groups)) % 5))
            f = grp.map(fold_of).values
            vals, p = [], np.zeros(len(T))
            for k in range(5):
                tr, te = np.where(f != k)[0], np.where(f == k)[0]
                m = gbm().fit(X_of(T.iloc[tr], feats), T.log_y.iloc[tr])
                p[te] = m.predict(X_of(T.iloc[te], feats))
                pi = permutation_importance(m, X_of(T.iloc[te], feats), T.log_y.iloc[te], n_repeats=3,
                                            random_state=SEED, scoring="r2")
                vals.append(pi.importances_mean)
            v = np.clip(np.mean(vals, axis=0), 0, None)
            reps.append(v / v.sum() if v.sum() > 0 else v)
            r2s.append(r2_score(T.log_y, p))
        A = pd.DataFrame(reps, columns=[LABELS[x] for x in feats])
        name = CITIES[c].split()[0]
        shares[name] = A
        cvr2[name] = (np.mean(r2s), np.std(r2s))
        rank = A.rank(axis=1, ascending=False)
        top[name] = pd.DataFrame({"Top1 유지율": (rank == 1).mean(), "Top3 유지율": (rank <= 3).mean()})
        say(f"  {CITIES[c]}: 도시 안 공간 CV R² {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    allf = list(dict.fromkeys(sum([list(a.columns) for a in shares.values()], [])))
    mean = pd.DataFrame({k: a.mean().reindex(allf) for k, a in shares.items()}) * 100
    std = pd.DataFrame({k: a.std().reindex(allf) for k, a in shares.items()}) * 100
    tops = pd.concat({k: v.reindex(allf) for k, v in top.items()}, axis=1)
    out = pd.concat({"비중평균(%)": mean, "비중표준편차(%p)": std}, axis=1).round(1).join(tops.round(2))
    out = out.sort_values(("비중평균(%)", "서울"), ascending=False)
    to_csv(out, "E3v2_importance_spatial.csv", index=True)
    to_csv(pd.DataFrame({k: {"도시 안 공간 CV R² 평균": v[0], "표준편차": v[1]} for k, v in cvr2.items()}).T.round(3).reset_index(),
           "E3v2_city_cv_r2.csv")
    for k in mean.columns:
        s = mean[k].sort_values(ascending=False).head(5)
        say(f"  {k} Top5: " + ", ".join(f"{i} {v:.0f}%±{std.loc[i, k]:.0f}" for i, v in s.items()))

    M = mean.loc[out.index]
    Sd = std.loc[out.index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(M) + 1.4))
    im = ax.imshow(M.fillna(0).values, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(range(3), M.columns)
    ax.set_yticks(range(len(M)), M.index)
    ax.xaxis.tick_top()
    for i in range(len(M)):
        for j in range(3):
            v, s = M.values[i, j], Sd.values[i, j]
            txt = "–" if np.isnan(v) else f"{v:.0f}% ±{s:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if (not np.isnan(v) and v > 15) else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, label="도시 안 상대적 중요도 비중 (%)")
    ax.set_title(f"도시별 상대적 변수 중요도 — 공간 그룹 CV {R_E3}회 반복 평균 ± 표준편차", loc="left", fontsize=11.5, pad=28)
    save(fig, "E3v2_importance_spatial.png")
    return out, cvr2


def e3_shap(df):
    say("\n[E3s] SHAP 세 도시 비교")
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        say("  lightgbm 미설치 → 생략")
        return None
    res = {}
    for c in CITIES:
        feats = FEATS_OF[c]
        T = df[df.city == c]
        X = X_of(T, feats)
        m = gbm().fit(X, T.log_y)
        sv = m.predict(X, pred_contrib=True)[:, :-1]
        imp = np.abs(sv).mean(0)
        dirn = [spearmanr(X.iloc[:, j], sv[:, j]).correlation if X.iloc[:, j].nunique() > 1 else 0 for j in range(len(feats))]
        res[CITIES[c].split()[0]] = pd.DataFrame({"평균|SHAP|": imp, "비중(%)": imp / imp.sum() * 100, "방향(값↑→예측)": dirn},
                                                 index=[LABELS[f] for f in feats])
    t = pd.concat(res, axis=1).round(3)
    to_csv(t, "E3s_shap_compare.csv", index=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (k, r) in zip(axes, res.items()):
        r = r.sort_values("비중(%)").tail(10)
        col = ["#c0392b" if d > 0.2 else "#2a6fdb" if d < -0.2 else "#9a9a9a" for d in r["방향(값↑→예측)"]]
        ax.barh(r.index, r["비중(%)"], color=col)
        ax.set_title(k, loc="left", fontsize=13)
        ax.set_xlabel("평균 |SHAP| 비중 (%)")
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color="#c0392b", label="값이 클수록 예측 이용량 ↑"),
                            Patch(color="#2a6fdb", label="값이 클수록 예측 이용량 ↓"),
                            Patch(color="#9a9a9a", label="방향 일정하지 않음")], frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("세 도시 모델의 SHAP 비교 — 상위 10개 변수 (각 도시 데이터로 학습한 모델)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, "E3s_shap_compare.png")
    imgs = [OLD_FIG / f"E3_shap_{c}.png" for c in CITIES]
    if all(p.exists() for p in imgs):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        for ax, p in zip(axes, imgs):
            ax.imshow(plt.imread(p))
            ax.axis("off")
        fig.tight_layout()
        save(fig, "E3s_shap_three_summary.png")
    return t


# ------------------------------------------------------------
def e4v2(df):
    say("\n[E4v2] 구별 오차 — log 기준 배율 vs 실제 건수 기준 배율")
    S = df[df.city == "seoul"].reset_index(drop=True)
    from sklearn.model_selection import GroupKFold
    p = np.zeros(len(S))
    for tr, te in GroupKFold(5).split(S, S.log_y, S.gu):
        p[te] = gbm().fit(X_of(S.iloc[tr], FULL), S.log_y.iloc[tr]).predict(X_of(S.iloc[te], FULL))
    parts = [S.assign(pred=p)]
    for c in ("daejeon", "changwon"):
        feats = FEATS_OF[c]
        T = df[df.city == c].reset_index(drop=True)
        parts.append(T.assign(pred=gbm().fit(X_of(S, feats), S.log_y).predict(X_of(T, feats))))
    d = pd.concat(parts)
    d["pred_cnt"] = np.expm1(d.pred).clip(lower=0)
    g = d.groupby(["city", "gu"]).apply(lambda t: pd.Series({
        "대여소 수": len(t),
        "log 기준 배율 (1+예측)/(1+실제) 기하평균": np.exp((t.pred - t.log_y).mean()),
        "건수 기준 배율 Σ예측/Σ실제": t.pred_cnt.sum() / t.daily_mean.sum(),
        "실제 하루 대여 중앙값": t.daily_mean.median()})).round(2).reset_index()
    city = d.groupby("city").apply(lambda t: pd.Series({
        "대여소 수": len(t),
        "log 기준 배율 (1+예측)/(1+실제) 기하평균": np.exp((t.pred - t.log_y).mean()),
        "건수 기준 배율 Σ예측/Σ실제": t.pred_cnt.sum() / t.daily_mean.sum(),
        "실제 하루 대여 중앙값": t.daily_mean.median()})).round(2).reset_index().assign(gu="(도시 전체)")
    g = pd.concat([city, g], ignore_index=True)
    to_csv(g, "E4v2_ratio_by_gu.csv")
    say(g[g.city != "seoul"].to_string(index=False))
    return d, g


def grade_map(d, gus):
    say("\n[G] 수요 등급 지도 (5분위, 1등급 = 상위 20%)")
    from viz_map import draw_gu, outline, extent, frame
    from matplotlib.patches import PathPatch
    rows = []
    cmap = ListedColormap(["#a50f15", "#ef6548", "#fdbb84", "#c6dbef", "#6b8fb3"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for i, c in enumerate(("daejeon", "changwon")):
        t = d[d.city == c].copy()
        t["g_pred"] = 5 - pd.qcut(t.pred.rank(method="first"), 5, labels=False)
        t["g_act"] = 5 - pd.qcut(t.log_y.rank(method="first"), 5, labels=False)
        diff = (t.g_pred - t.g_act).abs()
        top = t[t.g_pred == 1]
        rows.append({"도시": CITIES[c], "등급 일치율": (diff == 0).mean(), "±1등급 이내": (diff <= 1).mean(),
                     "예측 1등급 중 실제 1~2등급 비율": (top.g_act <= 2).mean(), "예측 5등급 중 실제 4~5등급 비율": (t[t.g_pred == 5].g_act >= 4).mean()})
        t[["city", "station", "gu", "lat", "lon", "daily_mean", "g_pred", "g_act"]].to_csv(OUT / f"G_grades_{c}.csv", index=False, encoding="utf-8-sig")
        for j, (col, lab) in enumerate((("g_pred", "서울 모델 예측 등급"), ("g_act", "실제 이용 등급"))):
            ax = axes[i, j]
            for g in gus[c]:
                ax.add_patch(PathPatch(g["path"], facecolor="#f4f4f4", edgecolor="none", zorder=0))
            outline(ax, gus[c])
            draw_gu(ax, gus[c], fs=8)
            tt = t.sort_values(col, ascending=False)
            sc = ax.scatter(tt.x, tt.y, c=tt[col], cmap=cmap, norm=norm, s=12, linewidths=0, zorder=4)
            frame(ax, extent(gus[c]))
            ax.set_title(f"{CITIES[c]} — {lab}", loc="left", fontsize=13)
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, ticks=[1, 2, 3, 4, 5])
    cb.ax.set_yticklabels(["1등급 (상위 20%)", "2등급", "3등급", "4등급", "5등급 (하위 20%)"])
    cb.ax.invert_yaxis()
    fig.suptitle("서울 모델이 매긴 수요 등급 vs 실제 이용 등급 — 예측 등급은 정책 우선순위가 아니라 검토 순서의 참고값임",
                 x=0.01, ha="left", fontsize=14)
    save(fig, "G_grade_map.png")
    t = pd.DataFrame(rows).round(3)
    to_csv(t, "G_grade_agreement.csv")
    say(t.to_string(index=False))
    return t


def hparams():
    m = gbm()
    keep = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "subsample_freq",
            "colsample_bytree", "random_state", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")
    rows = [{"항목": "GBM 구현", "값": type(m).__name__}]
    rows += [{"항목": f"GBM {k}", "값": v} for k, v in m.get_params().items() if k in keep]
    rows += [{"항목": "Ridge", "값": "StandardScaler + RidgeCV(alpha = 10^-2 ~ 10^3, 20개), 개수·거리 피처 log1p"},
             {"항목": "튜닝", "값": "하이퍼파라미터 고정(탐색 없음) — 전 실험 동일"},
             {"항목": "서울 내부 검증", "값": "자치구 GroupKFold 5-fold"},
             {"항목": "현지 공간 블록", "값": f"{BLOCK}m 격자 블록 단위 추출, 학습 대여소 {BUFFER}m 이내 평가 대여소 제외"},
             {"항목": "서울+현지 함께", "값": "현지 표본 가중치 5, 도시 구분 변수(is_local) 추가"},
             {"항목": "난수 시드", "값": f"{SEED} (+반복 번호)"},
             {"항목": "반복 횟수", "값": f"E2v2 {R_E2}회, E3v2 {R_E3}회, P1 {R_P1}회"}]
    try:
        import lightgbm, sklearn
        rows.append({"항목": "버전", "값": f"lightgbm {lightgbm.__version__}, scikit-learn {sklearn.__version__}, python {sys.version.split()[0]}"})
    except ImportError:
        import sklearn
        rows.append({"항목": "버전", "값": f"scikit-learn {sklearn.__version__}, python {sys.version.split()[0]} (lightgbm 없음)"})
    t = pd.DataFrame(rows)
    to_csv(t, "H_hyperparams.csv")
    return t


def main():
    font()
    say("모델:", type(gbm()).__name__, "| QUICK" if QUICK else "")
    df, gus = load()
    p0_counts(df)
    hparams()
    e1_final(df)
    e2v2(df)
    df = common_period_target(df)
    p1_period(df)
    e3v2(df)
    e3_shap(df)
    d, _ = e4v2(df)
    grade_map(d, gus)
    (OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    say("\n끝. 결과 요약:", OUT / "summary.txt")


if __name__ == "__main__":
    main()
