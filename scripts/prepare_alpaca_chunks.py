import json
import random
import re
from collections import defaultdict
from pathlib import Path

SOURCE = Path("data/parsed/paper_pages.jsonl")
OUT = Path("data/datasets")
OUT.mkdir(parents=True, exist_ok=True)


def normalize(text):
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.split(
        r"(?im)^\s*(?:references|bibliography)\s*$",
        text, maxsplit=1
    )[0]
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def cut_text(text, max_chars=900, min_chars=330):
    result = []

    while text:
        if len(text) <= max_chars:
            part, text = text.strip(), ""
        else:
            window = text[:max_chars]
            boundary = max(
                window.rfind(p, 560)
                for p in (". ", "; ", "? ", "! ")
            )

            if boundary < 560:
                boundary = window.rfind(" ", 650)

            boundary = (
                boundary + 1
                if boundary >= 450
                else max_chars
            )

            part = text[:boundary].strip()
            text = text[boundary:].strip()

        if len(part) >= min_chars:
            result.append(part)

    return result


chunks = []

with SOURCE.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        paper = row["paper_id"]
        page = int(row["page"])
        text = normalize(row["text"])

        for i, part in enumerate(cut_text(text), 1):
            chunks.append({
                "chunk_id": f"{paper}_p{page:02d}_c{i:02d}",
                "paper_id": paper,
                "page": page,
                "text": part,
            })

if not chunks:
    raise SystemExit("没有生成文本块，请检查PDF解析文件")

with (OUT / "all_chunks.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in chunks:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 每篇论文抽取3个不同页面的文本块
rng = random.Random(42)
pilots = []

for paper in ("YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"):
    by_page = defaultdict(list)

    for row in chunks:
        if row["paper_id"] == paper and row["page"] > 1:
            by_page[row["page"]].append(row)

    pages = sorted(by_page)
    rng.shuffle(pages)

    for pg in pages[:3]:
        pilots.append(rng.choice(by_page[pg]))

    count = sum(c["paper_id"] == paper for c in chunks)
    print(f"{paper}: {count} 个文本块")

with (OUT / "pilot_chunks.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in pilots:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with (OUT / "pilot_preview.txt").open(
    "w", encoding="utf-8"
) as f:
    for row in pilots:
        f.write(
            f'\n===== {row["chunk_id"]} =====\n'
            f'{row["text"]}\n'
        )

print(f"\n总文本块数: {len(chunks)}")
print(f"试验文本块数: {len(pilots)}")
print("文本分块完成!")
