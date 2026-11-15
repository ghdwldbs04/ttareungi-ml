# -*- coding: utf-8 -*-
"""
선행연구 PDF 다운로드 + PPT용 그림 캡처

    python paper_figures.py

결과:
    refs/pdf/<key>.pdf                원문 PDF
    refs/pages/<key>/p01.png ...       모든 페이지 PNG (그림을 못 잘랐을 때 여기서 직접 캡처)
    refs/figures/<key>/p03_그림2.png   캡션(그림/표/Fig/Table) 기준으로 자동으로 잘라낸 그림
    refs/figure_index.md               논문별 그림·표 캡션 목록 + 파일 경로 + 출처 표기

다운로드가 막힌 논문은 브라우저로 직접 받아서 refs/pdf/<key>.pdf 로 저장한 뒤 다시 실행.
수상작 PDF처럼 목록에 없는 PDF도 refs/pdf/ 에 넣으면 같이 처리됨 (파일명이 key가 됨).
"""
import re
import sys
import urllib.request
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF가 없음: pip install pymupdf")

ROOT = Path(__file__).resolve().parent / "refs"
PDF, PAGES, FIGS = ROOT / "pdf", ROOT / "pages", ROOT / "figures"
ZOOM_PAGE, ZOOM_FIG = 2.0, 3.0   # 페이지 약 144dpi, 그림 약 216dpi

PAPERS = [
    # key, 인용 표기, URL
    ("sa_lee_2018", "사경은·이수기 (2018). 서울시 공공자전거 이용에 영향을 미치는 물리적 환경 요인 분석. 국토계획 53(6).",
     "https://koreascience.kr/article/JAKO201829862639586.pdf"),
    ("kim_park_park_2020", "김예술·박시온·박건웅 (2020). 서울 공공자전거 신규 대여소를 위한 수요량 예측 분석. 응용통계연구 33(6).",
     "https://koreascience.kr/article/JAKO202010265883933.pdf"),
    ("ahn_lee_2022", "안민수·이창효 (2022). 공공자전거 이용에 영향을 미치는 공간 특성 분석: 대전 타슈.",
     "https://koreascience.kr/article/JAKO202209672004312.pdf"),
    ("lee_jeong_shin_2016", "이장호·정경옥·신희철 (2016). 기상조건과 입지특성이 공공자전거 이용에 미치는 영향 분석. 대한교통학회지 34(5).",
     "https://koreascience.kr/article/JAKO201606776006323.pdf"),
    ("shin_2016", "신상범 (2016). 한국 지방 도시 공공자전거 정책의 도입과 지속 요인: 창원시 누비자. 대한지리학회지 51(1).",
     "https://koreascience.kr/article/JAKO201612460483171.pdf"),
    ("todd_2021", "Todd, O'Brien, Cheshire (2021). A global comparison of bicycle sharing systems. J. Transport Geography.",
     "https://discovery.ucl.ac.uk/10130847/1/1-s2.0-S0966692321001721-main.pdf"),
    ("guidon_2020", "Guidon, Reck, Axhausen (2020). Expanding a(n) (electric) bicycle-sharing system to a new city.",
     "https://www.research-collection.ethz.ch/bitstream/20.500.11850/402553/1/v795.pdf"),
    ("guo_2022", "Guo, Yang, Chen (2022). Bike Share Usage and the Built Environment: A Review. Front. Public Health.",
     "https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.848169/pdf"),
    ("daegu_council_2024", "대구광역시의회 (2024.12). 탄소 감축을 위한 공유자전거 이용 활성화 방안 (정책연구보고서).",
     "https://council.daegu.go.kr/attach/bbs/policy/9809eeacfb137b2cf0c7217c2d72bb8a.pdf"),
    ("song_2026_tashu", "송현호·장하린·이상금 (2026). 타슈 공공자전거 수요 예측 및 패턴 분석에 관한 연구 (학술대회 초록).",
     "https://d2j16w31g89z0j.cloudfront.net/site/2026w/abs/0410-NSDDX.pdf"),
]

CAP = re.compile(r"^\s*[<\[〈(]?\s*(그\s*림|표|Fig(?:ure)?|FIG(?:URE)?|Table|TABLE)\s*\.?\s*(\d+)", re.I)


def download(key, url):
    out = PDF / f"{key}.pdf"
    if out.exists() and out.stat().st_size > 10_000:
        return "있음"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                                               "Accept": "application/pdf,*/*"})
    try:
        data = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        return f"실패 ({e.__class__.__name__}) -> 브라우저로 {url} 받아서 {out} 로 저장"
    if not data.startswith(b"%PDF"):
        return f"실패 (PDF가 아닌 페이지가 옴) -> 브라우저로 {url} 받아서 {out} 로 저장"
    out.write_bytes(data)
    return f"다운로드 {len(data) / 1e6:.1f}MB"


def safe(s):
    return re.sub(r"[^\w가-힣.-]+", "_", s)[:40].strip("_")


def crop_by_captions(doc, key):
    rows = []
    (FIGS / key).mkdir(parents=True, exist_ok=True)
    for pno, page in enumerate(doc, start=1):
        W, H = page.rect.width, page.rect.height
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        blocks.sort(key=lambda b: (b[1], b[0]))
        body = [b for b in blocks if len(b[4]) > 150]          # 본문 문단으로 볼 블록
        for b in blocks:
            m = CAP.match(b[4])
            if not m:
                continue
            kind = "표" if m.group(1).lower().startswith(("표", "table")) else "그림"
            num = m.group(2)
            caption = " ".join(b[4].split())[:120]
            x0, y0, x1, y1 = b[:4]
            wide = (x1 - x0) > W * 0.45 or (x0 < W * 0.3 and x1 > W * 0.7)
            left, right = (36, W - 36) if wide else ((36, W / 2 + 6) if x0 < W / 2 else (W / 2 - 6, W - 36))
            same_col = [t for t in body if t[2] > left and t[0] < right]
            if kind == "그림":   # 그림은 보통 캡션 위에 있음
                above = [t[3] for t in same_col if t[3] <= y0 - 2]
                top = max(above) if above else 40
                clip = fitz.Rect(left, top, right, y1 + 2)
            else:                # 표는 보통 캡션 아래에 있음
                below = [t[1] for t in same_col if t[1] >= y1 + 2]
                bottom = min(below) if below else H - 40
                clip = fitz.Rect(left, y0 - 2, right, bottom)
            if clip.height < 40:
                clip = fitz.Rect(left, max(40, y0 - H * 0.35), right, y1 + 2)   # 너무 작으면 위쪽 35%를 넉넉히
            name = f"p{pno:02d}_{kind}{num}.png"
            page.get_pixmap(matrix=fitz.Matrix(ZOOM_FIG, ZOOM_FIG), clip=clip & page.rect).save(FIGS / key / name)
            rows.append((pno, f"{kind} {num}", caption, f"figures/{key}/{name}"))
        # 캡션과 상관없이 PDF 안의 큰 이미지도 잘라 둠 (스캔 그림·지도 대비)
        try:
            for i, info in enumerate(page.get_image_info(), start=1):
                r = fitz.Rect(info["bbox"]) & page.rect
                if r.width * r.height > W * H * 0.04:
                    name = f"p{pno:02d}_image{i}.png"
                    page.get_pixmap(matrix=fitz.Matrix(ZOOM_FIG, ZOOM_FIG), clip=r).save(FIGS / key / name)
                    rows.append((pno, f"이미지 {i}", "(캡션 없이 추출된 이미지)", f"figures/{key}/{name}"))
        except Exception:
            pass
    return rows


def main():
    for d in (PDF, PAGES, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    cite = {k: c for k, c, _ in PAPERS}
    print("== 다운로드 ==")
    for key, _, url in PAPERS:
        print(f"  {key}: {download(key, url)}")

    print("\n== 그림 추출 ==")
    md = ["# 선행연구 그림·표 목록", "",
          "PPT에 쓸 때는 슬라이드 아래에 출처(저자, 연도, 그림 번호)를 꼭 적기.", ""]
    for pdf in sorted(PDF.glob("*.pdf")):
        key = pdf.stem
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"  {key}: 열기 실패 {e}")
            continue
        (PAGES / key).mkdir(parents=True, exist_ok=True)
        for pno, page in enumerate(doc, start=1):
            page.get_pixmap(matrix=fitz.Matrix(ZOOM_PAGE, ZOOM_PAGE)).save(PAGES / key / f"p{pno:02d}.png")
        rows = crop_by_captions(doc, key)
        print(f"  {key}: {len(doc)}쪽, 캡션/이미지 {len(rows)}개")
        md += [f"## {key}", "", f"출처: {cite.get(key, pdf.name)}", "", "| 쪽 | 번호 | 캡션 | 파일 |", "| --- | --- | --- | --- |"]
        md += [f"| {p} | {n} | {c.replace('|', '/')} | {f} |" for p, n, c, f in rows] or ["| - | - | 캡션을 못 찾음: pages 폴더에서 직접 캡처 | - |"]
        md.append("")
    (ROOT / "figure_index.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n목록 저장: {ROOT / 'figure_index.md'}")


if __name__ == "__main__":
    main()
