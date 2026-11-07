from pathlib import Path
from collections import Counter
import csv
import html
import json
import re

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data/papers"
OUT = ROOT / "results/milvus/visual_audit"

CAPTION = re.compile(
    r"^\s*(fig(?:ure)?|tab(?:le)?)\.?\s*\d+\b",
    re.I
)


def render_clip(page, rect, path, zoom=1.8):
    rect = pymupdf.Rect(rect) & page.rect
    if rect.is_empty or rect.width < 15 or rect.height < 15:
        return False

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        clip=rect,
        alpha=False
    )
    pix.save(str(path))
    return True


def find_captions(page):
    found = []

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            txt = "".join(
                span["text"]
                for span in line.get("spans", [])
            ).strip()

            match = CAPTION.match(txt)

            if match:
                kind = (
                    "caption_figure"
                    if match.group(1).lower().startswith("fig")
                    else "caption_table"
                )
                found.append((
                    txt[:240],
                    pymupdf.Rect(line["bbox"]),
                    kind
                ))

    return found


def main():
    pdfs = sorted(PDF_DIR.rglob("*.pdf"))

    if not pdfs:
        raise SystemExit("No PDFs in data/papers")

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    groups = []
    totals = Counter()

    for file_no, pdf in enumerate(pdfs, 1):
        folder = OUT / f"paper_{file_no:02d}"
        folder.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf)
        pages_html = []
        count = Counter()

        for page_idx, page in enumerate(doc):
            pno = page_idx + 1
            page_name = f"page_{pno:03d}.png"

            render_clip(
                page, page.rect,
                folder / page_name,
                zoom=1.05
            )

            candidates = []
            captions = find_captions(page)

            # Rough caption-guided crops.
            # Review carefully before using them.
            for caption_text, anchor, kind in captions:
                mid = page.rect.width / 2

                if anchor.x1 < mid + 15:
                    left = page.rect.x0 + 16
                    right = mid - 6
                elif anchor.x0 > mid - 15:
                    left = mid + 6
                    right = page.rect.x1 - 16
                else:
                    left = page.rect.x0 + 16
                    right = page.rect.x1 - 16

                if kind == "caption_figure":
                    top = anchor.y0 - 230
                    bottom = anchor.y0 - 4
                else:
                    top = anchor.y1 + 4
                    bottom = anchor.y1 + 195

                region = pymupdf.Rect(
                    left, top, right, bottom
                ) & page.rect

                if not region.is_empty and region.height > 35:
                    candidates.append((kind, region, ""))

            # Embedded raster images.
            seen = set()

            for item in page.get_image_info():
                box = (
                    pymupdf.Rect(item["bbox"])
                    & page.rect
                )

                if (
                    box.is_empty
                    or box.width < 65
                    or box.height < 50
                ):
                    continue

                if (
                    box.get_area()
                    < page.rect.get_area() * 0.018
                ):
                    continue

                key = tuple(round(v, 1) for v in box)

                if key not in seen:
                    seen.add(key)
                    candidates.append(("image", box, ""))

            # Automatically detected tables.
            try:
                finder = page.find_tables()

                for table in finder.tables:
                    box = (
                        pymupdf.Rect(table.bbox)
                        & page.rect
                    )

                    if (
                        not box.is_empty
                        and box.width > 60
                        and box.height > 28
                    ):
                        candidates.append((
                            "table",
                            box,
                            table.to_markdown()
                        ))
            except Exception as exc:
                print(
                    f"Table warning: {pdf.name} "
                    f"page {pno}: {exc}"
                )

            cards = []

            for seq, (kind, box, table_text) in enumerate(
                candidates, 1
            ):
                name = (
                    f"p{pno:03d}_"
                    f"{kind}_{seq:02d}.png"
                )

                if not render_clip(
                    page,
                    box + (-4, -4, 4, 4),
                    folder / name
                ):
                    continue

                if table_text:
                    markdown_name = name.replace(
                        ".png", ".md"
                    )
                    (folder / markdown_name).write_text(
                        table_text,
                        encoding="utf-8"
                    )

                rows.append({
                    "pdf_file": str(pdf.relative_to(ROOT)),
                    "pdf_page": pno,
                    "type": kind,
                    "image_file": str(
                        (folder / name).relative_to(OUT)
                    ),
                    "bbox": json.dumps([
                        round(v, 2) for v in box
                    ]),
                    "caption_candidates": " | ".join(
                        c[0] for c in captions
                    ),
                    "verified": "NO"
                })

                count[kind] += 1

                cards.append(
                    f'<a href="{name}">'
                    f'<img src="{name}" width="210">'
                    f'</a>'
                )

            caption_html = html.escape(
                " | ".join(c[0] for c in captions)
                or "(none)"
            )

            pages_html.append(
                f'<section><h3>PDF page {pno}</h3>'
                f'<p>Caption hints: {caption_html}</p>'
                f'<a href="{page_name}">'
                f'<img src="{page_name}" width="320">'
                f'</a><div>'
                + ("".join(cards) or "No automatic candidates")
                + "</div></section>"
            )

            count["pages"] += 1
            count["caption_lines"] += len(captions)

        doc.close()

        (folder / "index.html").write_text(
            '<meta charset="utf-8">'
            '<style>'
            'body{font-family:sans-serif;max-width:1000px;'
            'margin:30px auto}'
            'section{border-bottom:1px solid #ddd;'
            'padding:16px 0}'
            'img{object-fit:contain;margin:8px;'
            'border:1px solid #ddd}'
            'p{color:#555}'
            '</style>'
            f'<h1>{html.escape(pdf.name)}</h1>'
            + "".join(pages_html),
            encoding="utf-8"
        )

        groups.append((
            pdf.name,
            f"paper_{file_no:02d}/index.html",
            dict(count)
        ))

        totals.update(count)
        print(f"{pdf.name}: {dict(count)}")

    with (OUT / "visual_candidates.csv").open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_file",
                "pdf_page",
                "type",
                "image_file",
                "bbox",
                "caption_candidates",
                "verified"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    listing = "".join(
        f'<li><a href="{link}">'
        f'{html.escape(name)}</a> — '
        f'{html.escape(str(counts))}</li>'
        for name, link, counts in groups
    )

    (OUT / "index.html").write_text(
        '<meta charset="utf-8">'
        '<h1>YOLO PDF visual audit</h1>'
        '<p>Review candidate crops against original pages.</p>'
        f'<ul>{listing}</ul>',
        encoding="utf-8"
    )

    print("TOTAL:", dict(totals))
    print("Review gallery:", OUT / "index.html")
    print("Candidates CSV:", OUT / "visual_candidates.csv")


if __name__ == "__main__":
    main()
