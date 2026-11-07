#!/usr/bin/env python3
"""Curate 4 YOLO papers using manually reviewed arXiv page regions.

Based on visual inspections of public arXiv PDFs (2107.08430, 2207.02696,
2402.13616, 2405.14458). Crops must still be checked against the user's
local files: matching page counts alone does not prove binary identity.

Run: python scripts/yolo_visual_curate.py --project .
Optional: --include-supplement
"""

import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path

import pymupdf

EXPECTED_PAGES = {"YOLOX": 7, "YOLOv7": 15, "YOLOv9": 18, "YOLOv10": 21}

# rect = [left, top, right, bottom] as fractions of PDF page width/height.
# These are fixed, manually-selected regions, NOT captions-based guesses.
# grade A = essential to the question-answering demonstrations; grade B = optional.
SPECS = [
    ("YOLOX_F1", "YOLOX", 1, "figure", 1, "A", (.12,.245,.87,.515), "检测准确率与延迟、模型规模的比较曲线", "speed accuracy size"),
    ("YOLOX_F2", "YOLOX", 3, "figure", 2, "A", (.16,.083,.81,.445), "Coupled Head 与 Decoupled Head 结构对比", "decoupled head network"),
    ("YOLOX_T2", "YOLOX", 4, "table", 2, "A", (.195,.080,.765,.305), "YOLOX 逐步引入各项改进的消融实验（含 SimOTA）", "SimOTA anchor free ablation"),
    ("YOLOX_T3", "YOLOX", 5, "table", 3, "A", (.076,.082,.465,.334), "YOLOX 与 YOLOv5 的 AP、参数量和延迟比较", "AP comparison latency"),
    ("YOLOX_T6", "YOLOX", 6, "table", 6, "A", (.109,.084,.852,.465), "不同目标检测器在 COCO 上的总体性能比较", "COCO AP FPS comparison"),

    ("YOLOv7_F2", "YOLOv7", 3, "figure", 2, "A", (.072,.075,.882,.395), "VoVNet、CSPVoVNet、ELAN 与 E-ELAN 结构对比", "E-ELAN architecture"),
    ("YOLOv7_F3", "YOLOv7", 4, "figure", 3, "A", (.076,.085,.885,.294), "Concat 型网络的宽度和深度联合缩放方案", "compound scaling"),
    ("YOLOv7_F5", "YOLOv7", 5, "figure", 5, "A", (.074,.084,.890,.288), "辅助检测头与主检测头的标签分配设计", "label assignment auxiliary head"),
    ("YOLOv7_T1", "YOLOv7", 6, "table", 1, "A", (.149,.076,.821,.447), "YOLOv7 相比基础模型的参数量、FLOPs 和 AP", "baseline comparison"),
    ("YOLOv7_T2", "YOLOv7", 7, "table", 2, "A", (.081,.076,.881,.581), "YOLOv7 与主流实时目标检测模型性能对比", "SOTA COCO AP FPS"),

    ("YOLOv9_F3", "YOLOv9", 5, "figure", 3, "A", (.073,.082,.882,.427), "PGI 与 PAN、RevCol、Deep Supervision 的结构对比", "PGI architecture"),
    ("YOLOv9_F4", "YOLOv9", 6, "figure", 4, "A", (.146,.075,.810,.344), "CSPNet、ELAN 与 GELAN 网络结构图", "GELAN structure"),
    ("YOLOv9_F5", "YOLOv9", 8, "figure", 5, "A", (.076,.078,.862,.369), "YOLOv9、GELAN 等的准确率与参数/FLOPs 曲线", "AP parameters FLOPs"),
    ("YOLOv9_T1", "YOLOv9", 7, "table", 1, "A", (.100,.078,.859,.667), "YOLOv9 和其他 YOLO 系列的 COCO 指标对比", "SOTA COCO AP"),
    ("YOLOv9_T5", "YOLOv9", 9, "table", 5, "A", (.485,.082,.869,.318), "引入 PGI 或深度监督后的模型消融对比", "PGI ablation"),

    ("YOLOv10_F1", "YOLOv10", 1, "figure", 1, "A", (.211,.320,.789,.543), "YOLOv10 推理延迟、参数量与精度的性能曲线", "latency AP speed"),
    ("YOLOv10_F2", "YOLOv10", 4, "figure", 2, "A", (.169,.083,.807,.305), "NMS-free 的一致性双标签分配与双检测头", "NMS free consistent dual assignments"),
    ("YOLOv10_F3", "YOLOv10", 6, "figure", 3, "A", (.209,.077,.796,.370), "冗余度分析与 CIB、PSA 模块结构", "CIB PSA rank"),
    ("YOLOv10_T1", "YOLOv10", 7, "table", 1, "A", (.170,.080,.801,.606), "YOLOv10 与 SOTA 模型的 AP、FLOPs 和延迟比较", "COCO AP latency FLOPs"),
    ("YOLOv10_T2", "YOLOv10", 8, "table", 2, "A", (.182,.077,.795,.246), "YOLOv10 主要模块的消融实验", "NMS free ablation"),

    ("YOLOv7_F4", "YOLOv7", 4, "figure", 4, "B", (.490,.292,.880,.592), "计划重参数化结构对比（补充）", "planned reparameterization"),
    ("YOLOv9_F6", "YOLOv9", 10, "figure", 6, "B", (.072,.080,.875,.458), "不同模型深度下的特征图可视化（补充）", "feature map visualization"),
    ("YOLOv9_APP_T1", "YOLOv9", 15, "table", 1, "B", (.120,.155,.414,.544), "附录训练超参数，仅作事实核对（补充）", "training hyperparameters"),
    ("YOLOv10_F6", "YOLOv10", 20, "figure", 6, "B", (.178,.299,.795,.678), "复杂场景检测示例，仅作展示，不用作主要技术检索证据", "qualitative detection examples"),
]


def pdf_path(pdf_dir, paper):
    matches = [p for p in pdf_dir.rglob("*.pdf") if p.stem.casefold() == paper.casefold()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {paper}.pdf in {pdf_dir}; found {matches}")
    return matches[0]


def filehash(file):
    h = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_specs():
    specs = []
    for (identifier, paper, page, typ, number, grade, bbox, desc, tags) in SPECS:
        specs.append(dict(id=identifier, paper_id=paper, pdf_page=page, kind=typ,
                          figure_table_number=number, grade=grade, bbox_norm=list(bbox),
                          description=desc, tags=tags))
    return specs


def extract_pdf_caption(page, kind, number):
    """Extract a short literal caption from text blocks, not an LLM paraphrase."""
    label = "Figure" if kind == "figure" else "Table"
    pattern = re.compile(rf"\b{label}\s*\.?\s*{number}\s*[:.]", re.I)
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        lines = [line.strip() for line in str(block[4]).splitlines() if line.strip()]
        if not lines:
            continue
        joined = " ".join(lines)
        match = pattern.search(joined)
        if match and match.start() <= 55:
            return joined[match.start():match.start()+600]
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".", help="Project root containing data/papers")
    ap.add_argument("--include-supplement", action="store_true")
    ap.add_argument("--scale", type=float, default=4.0,
                    help="PDF pixel scale; 4.0 is about 288 dpi")
    args = ap.parse_args()
    if not 2 <= args.scale <= 5:
        ap.error("--scale must be within 2..5")

    root = Path(args.project).expanduser().resolve()
    pdf_dir = root / "data/papers"
    out = root / "results/milvus/visual_curated_v1"
    crops_dir = out / "assets"
    previews_dir = out / "fullpage_previews"
    crops_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    specs = normalize_specs()
    selected = [s for s in specs if s["grade"] == "A" or args.include_supplement]
    all_rows = []
    docs = {}
    paths = {}
    hashes = {}

    try:
        for paper, num in EXPECTED_PAGES.items():
            path = pdf_path(pdf_dir, paper)
            d = pymupdf.open(path)
            if len(d) != num:
                raise RuntimeError(f"{paper}: expected {num} PDF pages, found {len(d)}."
                                   " Crops would no longer be verified.")
            docs[paper] = d
            paths[paper] = str(path.relative_to(root))
            hashes[paper] = filehash(path)

        html_parts = ["<meta charset='utf-8'><title>YOLO Visual Curation</title>",
                      "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1150px; margin:25px auto; background:#f8f9fc;color:#14212c}",
                      "article{background:white;border:1px solid #ddd;border-radius:12px;padding:22px;margin:20px 0}",
                      "img{max-width:100%;background:white;outline:1px solid #ddd}p,small{line-height:1.6}",
                      "a{color:#146e7b}.tag{background:#e1f1ef;border-radius:5px;padding:4px 8px}</style>",
                      "<h1>YOLO curated figures and tables (public-PDF reference)</h1>",
                      "<p><strong>Important:</strong> these are manually chosen regions from public arXiv PDFs,"
                      " not a confirmed audit of your local gallery. Compare cropped content and captions with your local original PDF."
                      " Do not ingest B-grade supplementary content as core evidence.</p>"]

        preview_created = set()
        for s in selected:
            doc = docs[s["paper_id"]]
            page = doc[s["pdf_page"] - 1]
            w, h = page.rect.width, page.rect.height
            x0,y0,x1,y1 = s["bbox_norm"]
            rect = pymupdf.Rect(w*x0, h*y0, w*x1, h*y1) & page.rect
            if rect.is_empty or rect.width < 30 or rect.height < 25:
                raise RuntimeError(f"Invalid crop {s['id']}: {rect}")

            png = crops_dir / f"{s['id']}.png"
            pm = page.get_pixmap(matrix=pymupdf.Matrix(args.scale,args.scale),
                                    clip=rect, colorspace=pymupdf.csRGB, alpha=False)
            pm.save(png)

            pkey = (s["paper_id"],s["pdf_page"])
            preview = previews_dir / f"{s['paper_id']}_p{s['pdf_page']:02d}.png"
            if pkey not in preview_created:
                page.get_pixmap(matrix=pymupdf.Matrix(1.35,1.35),
                                colorspace=pymupdf.csRGB,alpha=False).save(preview)
                preview_created.add(pkey)

            original_caption = extract_pdf_caption(
                page, s["kind"], s["figure_table_number"]
            )
            caption_ok = bool(original_caption)
            flags = []
            if not caption_ok: flags.append("CAPTION_NOT_FOUND_IN_PDF_TEXT_BLOCKS")
            if pm.width < 700: flags.append("SMALL_CROP_WIDTH")
            if pm.height < 200: flags.append("SMALL_CROP_HEIGHT")

            # Approximate native DPI of embedded raster material overlapping this crop.
            raster_dpis = []
            for item in page.get_image_info():
                im_rect = pymupdf.Rect(item.get("bbox", (0,0,0,0)))
                if im_rect.is_empty: continue
                intersection = im_rect & rect
                if intersection.is_empty or intersection.get_area() / max(1, im_rect.get_area()) < .3:
                    continue
                nw = item.get("width") or 0
                if nw > 0 and im_rect.width > 0:
                    raster_dpis.append(round(nw * 72 / im_rect.width))
            min_dpi = min(raster_dpis) if raster_dpis else None
            if min_dpi is not None and min_dpi < 120:
                flags.append("NATIVE_RASTER_LOW_DPI_CHECK_VISUALLY")

            rec = dict(s)
            rec.update(source_pdf=paths[s["paper_id"]],
                       source_sha256=hashes[s["paper_id"]],
                       bbox_pdf_points=[round(v,2) for v in rect],
                       image_path=str(png.relative_to(root)),
                       page_preview_path=str(preview.relative_to(root)),
                       image_width=pm.width, image_height=pm.height,
                       source_caption=original_caption,
                       image_raster_min_dpi=min_dpi,
                       caption_found_on_local_page=caption_ok,
                       quality_flags=flags,
                       status="PUBLIC_PDF_PRESELECTED_LOCAL_VISUAL_QA_PENDING")
            all_rows.append(rec)

            image_rel = "assets/" + png.name
            preview_rel = "fullpage_previews/" + preview.name
            html_parts.append(
                f"<article><h2>{html.escape(s['id'])} · PDF p.{s['pdf_page']} "
                f"<span class='tag'>Grade {s['grade']}</span></h2>"
                f"<p>{html.escape(s['description'])}</p>"
                f"<p><small>Original PDF caption: {html.escape(original_caption or '[not extracted]')}</small></p>"
                f"<p><small>Caption check: {'OK' if caption_ok else 'REVIEW'}; "
                f"pixels: {pm.width}x{pm.height}; flags: {html.escape(str(flags))}</small></p>"
                f"<a href='{image_rel}'><img src='{image_rel}' loading='lazy'></a>"
                f"<p><a href='{preview_rel}'>Compare with complete source PDF page</a></p>"
                f"</article>"
            )
        (out / "index.html").write_text("\n".join(html_parts), encoding="utf-8")
        with (out / "curated_records.jsonl").open("w",encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row,ensure_ascii=False) + "\n")
        with (out / "curated_records.csv").open("w",encoding="utf-8",newline="") as f:
            fields = ["id","paper_id","pdf_page","kind","figure_table_number","grade",
                      "description","source_caption","image_path","source_pdf","source_sha256","image_width","image_height",
                      "image_raster_min_dpi","caption_found_on_local_page","quality_flags","status"]
            wr = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            wr.writeheader()
            for row in all_rows:
                wr.writerow({**row, "quality_flags": ";".join(row["quality_flags"])})

        print("Source PDFs:")
        for paper, p in paths.items(): print(" ",paper, p, hashes[paper][:12])
        print("Curated crops:",len(all_rows))
        print("Crops by paper:", {p:sum(r["paper_id"]==p for r in all_rows) for p in EXPECTED_PAGES})
        print("Flagged crops:",[(r["id"],r["quality_flags"]) for r in all_rows if r["quality_flags"]])
        print("Review gallery:",out / "index.html")
        print("Structured records:",out / "curated_records.jsonl")
        print("NOTE: these are proposed high-resolution crops, NOT yet approved for Milvus ingestion.")
    finally:
        for d in docs.values(): d.close()


if __name__ == "__main__":
    main()
