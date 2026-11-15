
from pathlib import Path
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
STATIONS = BASE / "processed" / "stations_clean.csv"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. rasterio 설치 확인")

try:
    import rasterio
    from rasterio.warp import transform
except ImportError:
    print("rasterio 패키지가 설치되지 않았습니다.")
    print("오류 내용을 그대로 보내주세요.")
    raise

print("rasterio 버전:", rasterio.__version__)

print("\n" + "=" * 70)
print("2. 전국 경사도 GeoTIFF 찾기")

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

SLOPE = slope_dirs[0]

candidates = [
    p for p in SLOPE.rglob("*.tif")
    if nfc(p.name).lower() == "slope_all.tif"
]

if len(candidates) != 1:
    raise ValueError(
        f"Slope_All.tif 파일 확인 필요: {len(candidates)}개"
    )

RASTER = candidates[0]

print("파일:", nfc(RASTER.relative_to(BASE)))
print(
    "파일 크기(MB):",
    round(RASTER.stat().st_size / 1024**2, 2)
)

print("\n" + "=" * 70)
print("3. 래스터 메타데이터")

with rasterio.open(RASTER) as src:
    print("드라이버:", src.driver)
    print("좌표계:", src.crs)
    print("가로 픽셀 수:", src.width)
    print("세로 픽셀 수:", src.height)
    print("밴드 수:", src.count)
    print("자료형:", src.dtypes)
    print("NoData:", src.nodata)
    print("해상도:", src.res)
    print("경계:", src.bounds)
    print("단위 정보:", src.units)
    print("설명:", src.descriptions)

    print("\n전체 래스터 메모리 적재 예상 크기:")
    bytes_per_pixel = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 2,
        "uint32": 4,
        "int32": 4,
        "float32": 4,
        "float64": 8,
    }

    estimated_bytes = (
        src.width
        * src.height
        * sum(
            bytes_per_pixel.get(dtype, 4)
            for dtype in src.dtypes
        )
    )

    print(
        round(estimated_bytes / 1024**3, 2),
        "GB"
    )

    print("\n" + "=" * 70)
    print("4. 도시별 대여소 좌표 샘플링")

    stations = pd.read_csv(
        STATIONS,
        dtype={"station": str},
        low_memory=False
    )

    samples = (
        stations.groupby(
            "city",
            sort=True,
            group_keys=False
        )
        .head(5)
        .copy()
    )

    print("샘플 대여소 수:", len(samples))

    if src.crs is None:
        raise ValueError(
            "래스터 좌표계 정보가 없습니다."
        )

    xs, ys = transform(
        "EPSG:4326",
        src.crs,
        samples["lon"].tolist(),
        samples["lat"].tolist()
    )

    coords = list(zip(xs, ys))

    inside = [
        (
            src.bounds.left <= x < src.bounds.right
            and src.bounds.bottom < y <= src.bounds.top
        )
        for x, y in coords
    ]

    # masked=True:
    # NoData 값은 결측으로 취급
    values = list(
        src.sample(
            coords,
            indexes=1,
            masked=True
        )
    )

    sampled_values = []

    for value in values:
        item = value[0]

        if bool(getattr(item, "mask", False)):
            sampled_values.append(None)
        else:
            sampled_values.append(float(item))

    samples["inside_raster"] = inside
    samples["slope_sample"] = sampled_values

    print(
        samples[
            [
                "city",
                "station",
                "lat",
                "lon",
                "inside_raster",
                "slope_sample",
            ]
        ].to_string(index=False)
    )

    print("\n도시별 샘플 결측 수:")
    print(
        samples.groupby("city")["slope_sample"]
        .apply(lambda x: x.isna().sum())
        .to_string()
    )

print("\n경사도 래스터 구조 검사 완료!")
