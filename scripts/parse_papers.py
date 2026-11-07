import json
from pathlib import Path
import pymupdf

PAPER_DIR = Path("data/papers")
OUTPUT_DIR = Path("data/parsed")
PREVIEW_DIR = Path("data/previews")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

papers = {
    "YOLOX": "2107.08430",
    "YOLOv7": "2207.02696",
    "YOLOv9": "2402.13616",
    "YOLOv10": "2405.14458",
}

output_file = OUTPUT_DIR / "paper_pages.jsonl"
total_pages = 0

with output_file.open("w", encoding="utf-8") as fout:
    for paper_id, arxiv_id in papers.items():
        pdf_path = PAPER_DIR / f"{paper_id}.pdf"

        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        doc = pymupdf.open(pdf_path)
        if doc.page_count < 2:
            raise ValueError(f"PDF页数异常: {pdf_path}")

        paper_text = []
        extracted_pages = 0

        for page_index, page in enumerate(doc):
            text = page.get_text("text", sort=True).strip()

            if not text:
                continue

            blocks = []
            for b in page.get_text("blocks"):
                if b[6] == 0 and b[4].strip():
                    blocks.append({
                        "bbox": [round(v, 2) for v in b[:4]],
                        "text": b[4].strip()
                    })

            record = {
                "paper_id": paper_id,
                "arxiv_id": arxiv_id,
                "page": page_index + 1,
                "text": text,
                "blocks": blocks
            }

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            paper_text.append(
                f"\n===== PDF PAGE {page_index + 1} =====\n{text}"
            )
            extracted_pages += 1

        # 单独保存每篇论文的完整文本
        (OUTPUT_DIR / f"{paper_id}.txt").write_text(
            "\n".join(paper_text),
            encoding="utf-8"
        )

        # 渲染第一页，供人工检查 PDF 是否正确
        doc[0].get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False
        ).save(PREVIEW_DIR / f"{paper_id}_page1.png")

        total_pages += extracted_pages

        print(
            f"{paper_id}: PDF共{doc.page_count}页，"
            f"提取文本{extracted_pages}页"
        )

        doc.close()

print(f"\n总共提取文本页数: {total_pages}")
print(f"结构化数据保存至: {output_file}")
print("论文解析完成!")
