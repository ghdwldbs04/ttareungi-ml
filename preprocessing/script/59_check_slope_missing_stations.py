
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
STATIONS = BASE / "processed" / "stations_clean.csv"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 경사도 파일과 대여소 읽기")

slope_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and (
        "Slope" in nfc(p.name)
        or "경사" in nfc(p.name)
    )
]

if len(slope_dirs) != 1:
    raise ValueError("경사도 폴더 확인 필요")

candidates = [
    p for p in slope_dirs[0].rglob("*.tif")
    if nfc(p.name).lower() == "slope_all.tif"
]

if len(candidates) != 1:
    raise ValueError("Slope_All.tif 확인 필요")

RASTER = candidates[0]

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

print("전체 대여소:", len(stations))

print("\n" + "=" * 70)
print("2. 경사도 결측 대여소 찾기")

with rasterio.open(RASTER) as src:
    xs, ys = transform(
        "EPSG:4326",
        src.crs,
        stations["lon"].tolist(),
        stations["lat"].tolist()
    )

    coords = list(zip(xs, ys))

    sampled = list(
        src.sample(
            coords,
            indexes=1,
            masked=True
        )
    )

    values = []

    for item in sampled:
        value = item[0]

        if np.ma.is_masked(value):
            values.append(np.nan)
        else:
            values.append(float(value))

    stations["slope_value"] = values

    missing = stations[
        stations["slope_value"].isna()
    ].copy()

    print("결측 대여소 수:", len(missing))

    print(
        missing[
            ["city", "station", "lat", "lon"]
        ].to_string(index=False)
    )

    if len(missing) != 7:
        raise ValueError(
            "58번과 결측 대여소 수가 다릅니다."
        )

    print("\n" + "=" * 70)
    print("3. 결측 대여소 주변 5×5픽셀 조사")

    for _, station in missing.iterrows():
        x, y = transform(
            "EPSG:4326",
            src.crs,
            [station["lon"]],
            [station["lat"]]
        )

        row, col = src.index(x[0], y[0])

        window = Window(
            col_off=col - 2,
            row_off=row - 2,
            width=5,
            height=5
        )

        neighborhood = src.read(
            1,
            window=window,
            boundless=True,
            masked=True
        )

        valid = neighborhood.compressed()
        valid = valid[np.isfinite(valid)]

        print("\n" + "-" * 60)
        print(
            "대여소:",
            station["city"],
            station["station"]
        )

        print(
            "위도/경도:",
            station["lat"],
            station["lon"]
        )

        print("래스터 행/열:", row, col)
        print("주변 픽셀 수:", neighborhood.size)
        print("유효 픽셀 수:", len(valid))
        print(
            "결측 픽셀 수:",
            neighborhood.size - len(valid)
        )

        if len(valid) > 0:
            print(
                "주변 최솟값:",
                round(float(valid.min()), 4)
            )
            print(
                "주변 평균:",
                round(float(valid.mean()), 4)
            )
            print(
                "주변 최댓값:",
                round(float(valid.max()), 4)
            )
        else:
            print("주변 5×5픽셀 모두 결측")

        print("\n주변 경사도 배열:")
        print(
            np.array2string(
                neighborhood.filled(np.nan),
                precision=2,
                suppress_small=True
            )
        )

print("\n경사도 결측 대여소 조사 완료!")
