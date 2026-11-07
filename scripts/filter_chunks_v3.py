import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA = Path("data/datasets")
SOURCE = DATA / "all_chunks_v2.jsonl"

rows = [
    json.loads(line)
    for line in SOURCE.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

# 识别参考文献条目，例如 - [18] Author
REF_PATTERN = re.compile(
    r"(?:^|\s)-\s*\[\d{1,3}\]\s+[A-Z][A-Za-z-]+"
)

# 检测页面是否进入参考文献区域
page_text = defaultdict(list)

for row in rows:
    key = (row["paper_id"], row["page"])
    page_text[key].append(row["text"])

ref_start = {}

for (paper, page), parts in page_text.items():
    text = " ".join(parts)

    # 一页出现至少两个参考文献条目，
    # 视为已进入参考文献区域
    if len(REF_PATTERN.findall(text)) >= 2:
        ref_start[paper] = min(
            ref_start.get(paper, page),
            page
        )

print("检测到的参考文献起始页:", ref_start)

clean_rows = []
reasons = Counter()

for row in rows:
    text = row["text"].strip()
    paper = row["paper_id"]
    page = row["page"]

    # 排除参考文献页及之后的页面
    if paper in ref_start and page >= ref_start[paper]:
        reasons["reference_page"] += 1
        continue

    # 排除图片OCR残留、HTML及上标碎片
    if re.search(
        r"<!--|</?(?:sup|sub|br|span|img)\b",
        text,
        re.IGNORECASE
    ):
        reasons["html_or_picture"] += 1
        continue

    # 排除参考文献条目
    if REF_PATTERN.search(text):
        reasons["reference_entry"] += 1
        continue

    # 排除明显不完整的句子开头
    if re.match(r"^[a-z]", text):
        reasons["incomplete_start"] += 1
        continue

    if "\ufffd" in text:
        reasons["encoding_error"] += 1
        continue

    if not 260 <= len(text) <= 1000:
        reasons["length"] += 1
        continue

    # 排除疑似密集参考文献列表
    if len(re.findall(r"\[\d+\]", text)) > 8:
        reasons["too_many_citations"] += 1
        continue

    clean_rows.append(row)

# 保存全部候选正文
with (DATA / "all_chunks_v3.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in clean_rows:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

# 按不同页面选取12个试验片段
rng = random.Random(42)
pilot = []

keywords = [
    "YOLO", "COCO", "PGI", "GELAN",
    "anchor", "RepConv", "assignment",
    "performance", "accuracy", "training",
    "experiment", "detection"
]

for paper in ["YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"]:
    by_page = defaultdict(list)

    for row in clean_rows:
        if row["paper_id"] == paper and row["page"] > 1:
            by_page[row["page"]].append(row)

    pages = sorted(by_page)
    rng.shuffle(pages)

    for page in pages[:3]:
        candidates = by_page[page]

        # 优先选择技术信息更丰富的片段
        def score(row):
            text = row["text"].lower()
            return sum(
                word.lower() in text
                for word in keywords
            )

        pilot.append(max(candidates, key=score))

with (DATA / "pilot_chunks_v3.jsonl").open(
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
        f"{row['text']}"
    )

(DATA / "pilot_preview_v3.txt").write_text(
    "\n".join(preview),
    encoding="utf-8"
)

print("\n原始V2文本块:", len(rows))
print("保留V3文本块:", len(clean_rows))
print("过滤原因:", dict(reasons))
print("V3试验片段:", len(pilot))

for paper in ["YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"]:
    count = sum(
        r["paper_id"] == paper
        for r in clean_rows
    )
    print(f"{paper}: {count} 个候选正文块")

print("\nV3数据清洗完成!")
