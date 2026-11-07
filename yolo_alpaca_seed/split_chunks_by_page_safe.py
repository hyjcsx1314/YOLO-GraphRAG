#!/usr/bin/env python3
# 按论文+PDF页码隔离数据集；保留 V3 中的完整技术正文块。
# 不因为一句致谢或一条实验设置说明，就删除整块文本。
import json,random
from pathlib import Path
from collections import defaultdict

DATA=Path("data/datasets")
src=DATA/"all_chunks_v3.jsonl"
if not src.exists():
    raise SystemExit(f"找不到 {src}，请先完成 filter_chunks_v3.py")

rows=[json.loads(s) for s in src.read_text(encoding="utf-8").splitlines() if s.strip()]
rng=random.Random(42)
papers=defaultdict(set)
for r in rows:
    papers[r["paper_id"]].add(int(r["page"]))

manifest={}
assignment={}
for paper, pageset in sorted(papers.items()):
    pages=sorted(pageset)
    rng.shuffle(pages)
    n=len(pages)
    if n<3:
        raise SystemExit(f"{paper} 仅有 {n} 个来源页面，不能进行三个集合的页面级隔离")
    ntest=max(1,round(n*0.15))
    nval=max(1,round(n*0.15))
    if ntest+nval>=n: ntest=nval=1
    manifest[paper]={
        "test":sorted(pages[:ntest]),
        "val":sorted(pages[ntest:ntest+nval]),
        "train":sorted(pages[ntest+nval:])
    }
    for split in ("train","val","test"):
        for page in manifest[paper][split]:
            assignment[(paper,page)]=split

result={"train":[],"val":[],"test":[]}
for r in rows:
    result[assignment[(r["paper_id"],int(r["page"]))]].append(r)

for split,items in result.items():
    dest=DATA/f"{split}_chunks_v3.jsonl"
    dest.write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),
        encoding="utf-8"
    )
    print(f"{split}: {len(items)} 个文本块，{len({(r['paper_id'],r['page']) for r in items})} 个来源页面")

(DATA/"split_manifest_v3.json").write_text(
    json.dumps(manifest,ensure_ascii=False,indent=2),
    encoding="utf-8"
)
assert sum(map(len,result.values()))==len(rows)
print("页面隔离成功；全部",len(rows),"个 V3 文本块均已分配。")
