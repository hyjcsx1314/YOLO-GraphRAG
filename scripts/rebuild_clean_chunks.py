import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pymupdf
import pymupdf4llm

PAPERS = ["YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"]
OUT = Path("data/datasets")
MD_DIR = Path("data/parsed_layout")

OUT.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


def normalize(text):
    text = text.replace("\u00ad", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_paragraphs(markdown):
    """提取正文段落，排除明显的表格和参考文献。"""
    paragraphs = []
    buffer = []

    def flush():
        if buffer:
            text = normalize(" ".join(buffer))

            digit_ratio = (
                sum(c.isdigit() for c in text)
                / max(len(text), 1)
            )

            if len(text) >= 100 and digit_ratio < 0.18:
                paragraphs.append(text)

            buffer.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        heading = re.sub(
            r"^[#*\s]+", "", line
        ).strip().rstrip(":")

        if re.fullmatch(
            r"(?:\d+(?:\.\d+)*\s+)?"
            r"(?:references|bibliography)",
            heading,
            re.IGNORECASE
        ):
            flush()
            return paragraphs, True

        if not line or line.startswith("#"):
            flush()
            continue

        # 排除Markdown表格、图像标记
        if (
            line.startswith(("|", "![", "```"))
            or line.count("|") >= 2
        ):
            flush()
            continue

        # 排除图表标题及说明段首
        if re.match(
            r"(?i)^(?:fig(?:ure)?\.?|table)"
            r"\s*\d+[\s:.\-]",
            heading
        ):
            flush()
            continue

        # 排除明显的密集数值表格行
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b", line
        )
        if len(numbers) > 7:
            flush()
            continue

        buffer.append(line)

    flush()
    return paragraphs, False


def split_long(text):
    """尽量沿句子或空格切分长文本。"""
    while len(text) > 900:
        cut = text.rfind(". ", 400, 880)

        if cut < 400:
            cut = text.rfind(" ", 400, 880)

        if cut < 400:
            cut = 880

        yield text[:cut + 1].strip()
        text = text[cut + 1:].strip()

    if text:
        yield text


def build_chunks(paragraphs):
    chunks = []
    buffer = ""

    for paragraph in paragraphs:
        for segment in split_long(paragraph):
            if len(buffer) + len(segment) + 1 > 900:
                if len(buffer) >= 260:
                    chunks.append(buffer)
                buffer = segment
            else:
                buffer = (
                    buffer + " " + segment
                ).strip()

    if len(buffer) >= 260:
        chunks.append(buffer)

    return chunks


all_chunks = []
all_previews = []

for paper in PAPERS:
    path = Path("data/papers") / f"{paper}.pdf"

    with pymupdf.open(path) as doc:
        # 优先一次性按页提取
        data = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            ignore_images=True
        )

        if isinstance(data, list):
            pages = [item["text"] for item in data]
        else:
            # 兼容部分版本的按页输出行为
            pages = []
            for i in range(doc.page_count):
                result = pymupdf4llm.to_markdown(
                    doc,
                    pages=[i],
                    ignore_images=True
                )
                if isinstance(result, list):
                    result = result[0]["text"]
                pages.append(result)

    count = 0

    for page_number, markdown in enumerate(pages, 1):
        all_previews.append(
            f"\n===== {paper} PAGE {page_number} =====\n"
            + markdown
        )

        paragraphs, reached_refs = clean_paragraphs(
            markdown
        )

        chunks = build_chunks(paragraphs)

        for index, text in enumerate(chunks, 1):
            all_chunks.append({
                "chunk_id": (
                    f"{paper}_p{page_number:02d}"
                    f"_v2c{index:02d}"
                ),
                "paper_id": paper,
                "page": page_number,
                "text": text
            })
            count += 1

        if reached_refs:
            break

    print(f"{paper}: {count} 个正文文本块")

# 保存所有干净的文本块
with (OUT / "all_chunks_v2.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in all_chunks:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

(MD_DIR / "all_papers_preview.md").write_text(
    "\n".join(all_previews),
    encoding="utf-8"
)

# 每篇论文选择最多3个不同页面
pilot = []
rng = random.Random(42)

for paper in PAPERS:
    by_page = defaultdict(list)

    for row in all_chunks:
        if row["paper_id"] == paper and row["page"] > 1:
            by_page[row["page"]].append(row)

    pages = sorted(by_page)
    rng.shuffle(pages)

    for page in pages[:3]:
        pilot.append(rng.choice(by_page[page]))

with (OUT / "pilot_chunks_v2.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in pilot:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

preview = []

for row in pilot:
    preview.append(
        f"\n===== {row['chunk_id']} =====\n"
        + row["text"]
    )

(OUT / "pilot_preview_v2.txt").write_text(
    "\n".join(preview),
    encoding="utf-8"
)

print("\n优化后总文本块数:", len(all_chunks))
print("新试验文本块数:", len(pilot))
print("优化后的论文解析完成!")
