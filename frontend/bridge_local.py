#!/usr/bin/env python3
"""使用原项目真实 CLI 接入 Streamlit，不改写原有 RAG、模型、数据库。

stdin: {"question": "...", "mode": "GraphRAG|Vector RAG",
        "with_images": false, "top_k": 5}
stdout: 单个 JSON 对象 {"answer": "...", "evidence": [...], "meta": {...}}

实际问答由 scripts/ 中的后端程序产生，本模块仅转发调用并解析输出。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATHS = {
    "GraphRAG": Path("results/neo4j/graphrag_v1"),
    "Vector RAG": Path("results/milvus/multimodal_rag"),
}
SCRIPT_NAMES = {
    "GraphRAG": "yolo_graph_rag.py",
    "Vector RAG": "yolo_multimodal_rag.py",
}
MAX_JSON_BYTES = 20_000_000
MAX_QUESTION_LENGTH = 4000
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
ASSET_RE = re.compile(r"\b(YOLOX|YOLOv7|YOLOv9|YOLOv10)_[FT]\d+\b", re.I)
CITATION_RE = re.compile(r"\[([GTF]\d+)\]")


class IntegrationError(RuntimeError):
    """可安全提示用户的联调失败，不包含数据库或云端密钥。"""


def _str(value: Any, limit: int = 5000) -> str:
    return str(value or "")[:limit]


def _asset_id(value: Any) -> str | None:
    match = ASSET_RE.search(str(value or ""))
    return match.group(0) if match else None


def _path_within_project(path: Path, root: Path) -> Path | None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    data_dir = (resolved_root / "data").resolve()
    if not resolved.is_relative_to(data_dir):
        return None
    if resolved.suffix.lower() not in IMAGE_EXTS or not resolved.is_file():
        return None
    return resolved


def _image_path(
    root: Path,
    asset_id: str | None,
    raw_relative: Any = None,
    *,
    sent: bool = False,
    actual_sent_path: Any = None,
) -> str | None:
    """对展示图像查找真实项目文件；vision override 仅适用于实际发送图像。"""
    base = root.resolve()
    candidates: list[Path] = []

    if sent and actual_sent_path:
        selected = Path(str(actual_sent_path))
        candidates.append(selected if selected.is_absolute() else base / selected)
    if sent and asset_id:
        candidates.append(base / "data" / "visual_overrides" / f"{asset_id}.png")

    if isinstance(raw_relative, str) and raw_relative:
        raw = Path(raw_relative)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([
                base / "data" / "yolo_visual_final_v2" / raw,
                base / "data" / "yolo_visual_final_v2" / "assets" / raw,
                base / raw,
            ])

    if asset_id:
        for ext in IMAGE_EXTS:
            candidates.extend([
                base / "data" / "yolo_visual_final_v2" / "assets" / f"{asset_id}{ext}",
                base / "data" / "yolo_visual_final_v2" / f"{asset_id}{ext}",
            ])
    for candidate in candidates:
        valid = _path_within_project(candidate, base)
        if valid is not None:
            return valid.relative_to(base).as_posix()
    return None


def _source_pages(evidences: Any) -> str:
    if not isinstance(evidences, list):
        return ""
    lines = []
    for source in evidences[:4]:
        if isinstance(source, dict):
            paper = source.get("paper") or "Unknown"
            page = source.get("page")
            location = f"{paper} PDF p.{page}" if page is not None else str(paper)
            quote = _str(source.get("quote"), 300)
            lines.append(location + (f" | {quote}" if quote else ""))
    return "\n".join(lines)


def _graph_evidence(result: dict, root: Path) -> list[dict]:
    evidence: list[dict] = []
    sent_images = {
        item.get("asset_id"): item.get("path")
        for item in result.get("vision_images", [])
        if isinstance(item, dict) and item.get("asset_id")
    }
    sent_ids = set(result.get("vision_image_ids") or [])
    for fact in result.get("graph_facts") or []:
        if not isinstance(fact, dict):
            continue
        evidence.append(
            {
                "kind": "G",
                "id": _str(fact.get("citation_id"), 20),
                "title": f"{_str(fact.get('head'), 120)} ─{_str(fact.get('relation'), 80)}→ {_str(fact.get('tail'), 120)}",
                "snippet": _source_pages(fact.get("evidence")),
            }
        )
    m = result.get("milvus_evidence") or {}
    for item in m.get("text") or []:
        if isinstance(item, dict):
            evidence.append(
                {
                    "kind": "T",
                    "id": _str(item.get("id"), 20),
                    "title": _str(item.get("header"), 240),
                    "snippet": _str(item.get("snippet"), 4000),
                }
            )
    for item in m.get("figure") or []:
        if not isinstance(item, dict):
            continue
        asset = _asset_id(item.get("header"))
        sent = asset in sent_ids if asset else False
        evidence.append(
            {
                "kind": "F",
                "id": _str(item.get("id"), 20),
                "title": _str(item.get("header"), 240),
                "snippet": _str(item.get("snippet"), 1600),
                "asset_id": asset,
                "image_path": _image_path(root, asset, sent=sent, actual_sent_path=sent_images.get(asset)),
                "sent_to_model": sent,
            }
        )
    return evidence


def _vector_evidence(result: dict, stdout: str, root: Path) -> list[dict]:
    evidence: list[dict] = []
    sent_ids = set(re.findall(
        r"(?m)^VISION IMAGE SOURCE:\s*(\S+)\s*->",
        stdout,
    ))
    for item in result.get("text_evidence") or []:
        if not isinstance(item, dict):
            continue
        paper = _str(item.get("paper_id"), 32)
        page = item.get("pdf_page")
        citation_id = _str(item.get("citation_id"), 20)
        evidence.append(
            {
                "kind": "T",
                "id": citation_id,
                "title": f"{paper} · PDF p.{page} · {item.get('source')}",
                "snippet": _str(item.get("text"), 4000),
                "paper": paper,
                "page": page,
                "score": item.get("score"),
            }
        )
    for item in result.get("figure_evidence") or []:
        if not isinstance(item, dict):
            continue
        asset = _asset_id(item.get("asset_id"))
        sent = asset in sent_ids if asset else False
        paper = _str(item.get("paper_id"), 32)
        page = item.get("pdf_page")
        notes = [
            _str(item.get("description_zh"), 1400),
            _str(item.get("caption_pdf"), 1400),
        ]
        evidence.append(
            {
                "kind": "F",
                "id": _str(item.get("citation_id"), 20),
                "title": f"{asset or 'Figure'} · {paper} · PDF p.{page} · {_str(item.get('label'), 150)}",
                "snippet": "\n".join(n for n in notes if n),
                "paper": paper,
                "page": page,
                "score": item.get("rrf_score"),
                "asset_id": asset,
                "image_path": _image_path(
                    root, asset, item.get("image_path"), sent=sent
                ),
                "sent_to_model": sent,
            }
        )
    return evidence


def _result_file(stdout: str, root: Path, mode: str) -> Path:
    matches = re.findall(r"(?m)^Result JSON:\s*(.+?)\s*$", stdout)
    if not matches:
        raise IntegrationError(
            "真实后端没有输出 Result JSON 路径，请先用原脚本独立测试。"
        )
    selected = Path(matches[-1].strip())
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    folder = (root / RESULT_PATHS[mode]).resolve()
    if not selected.is_relative_to(folder) or selected.suffix.lower() != ".json":
        raise IntegrationError("结果 JSON 路径不在本项目允许的 results 子目录内。")
    if not selected.is_file():
        raise IntegrationError("原脚本提示结果已保存，但对应 JSON 文件不存在。")
    if selected.stat().st_size > MAX_JSON_BYTES:
        raise IntegrationError("结果 JSON 体积异常，拒绝加载。")
    return selected


def _read_result_file(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError("无法解析真实后端保存的 JSON。") from exc
    if not isinstance(value, dict):
        raise IntegrationError("真实后端保存的 JSON 不是对象。")
    return value


def _command(request: dict, root: Path) -> list[str]:
    mode = request["mode"]
    script = root / "scripts" / SCRIPT_NAMES[mode]
    if not script.is_file():
        raise IntegrationError(
            f"项目缺少 scripts/{SCRIPT_NAMES[mode]}；请将前端放在原项目根目录。"
        )
    argv = [
        sys.executable, str(script), "ask", "--question", request["question"], "--mode", "cloud",
    ]
    if mode == "GraphRAG":
        argv.extend(["--hops", "2"])
    else:
        # GraphRAG 现有源码固定检索 6 段文本与 5 张图表；不伪称 slider 生效。
        argv.extend([
            "--text-k", str(request["top_k"]),
            "--figure-k", str(request["top_k"]),
            "--routing", "auto",
        ])
    if request["with_images"]:
        argv.extend(["--vision", "--vision-count", "2"])
    return argv


def _validate_request(request: Any) -> dict:
    if not isinstance(request, dict):
        raise IntegrationError("请求必须是 JSON 对象。")
    question = request.get("question")
    mode = request.get("mode")
    with_images = request.get("with_images", False)
    top_k = request.get("top_k", 5)
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_LENGTH:
        raise IntegrationError("问题不能为空且不能超过 4000 字。")
    if mode not in RESULT_PATHS:
        raise IntegrationError("问答模式必须是 GraphRAG 或 Vector RAG。")
    if not isinstance(with_images, bool):
        raise IntegrationError("with_images 必须是布尔值。")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
        raise IntegrationError("top_k 必须是 1 至 10 的整数。")
    return {
        "question": question.strip(),
        "mode": mode,
        "with_images": with_images,
        "top_k": top_k,
    }


def query_existing_project(request: dict, root: Path = ROOT) -> dict:
    parsed = _validate_request(request)
    root = root.resolve()
    argv = _command(parsed, root)

    # 外层 app 的默认超时 900 秒；内层只给现有脚本 840 秒。
    try:
        run = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=int(os.getenv("YOLO_UI_SCRIPT_TIMEOUT", "840")),
        )
    except subprocess.TimeoutExpired as exc:
        raise IntegrationError("真实后端运行超时；请检查 Milvus、Neo4j 和模型接口。") from exc
    except OSError as exc:
        raise IntegrationError("无法启动真实后端 Python 脚本。") from exc

    if run.returncode != 0:
        # 不在用户界面或 STDOUT 原样泄露 stderr（可能带有私密信息）。
        print(
            f"[YOLO UI] Backend {parsed['mode']} exit={run.returncode}. "
            f"Diagnose original CLI in terminal. Last stderr lines (private terminal only):",
            file=sys.stderr,
        )
        print("\n".join(run.stderr.splitlines()[-8:])[:3000], file=sys.stderr)
        raise IntegrationError(
            f"{parsed['mode']} 原有脚本调用失败（退出码 {run.returncode}）。"
            "请检查当前终端中的环境变量、Milvus/Neo4j 服务以及 API 配额。"
        )

    path = _result_file(run.stdout, root, parsed["mode"])
    result = _read_result_file(path)
    if result.get("question") != parsed["question"] or result.get("mode") != "cloud":
        raise IntegrationError("本次结果 JSON 的问题或模式与网页请求不一致。")

    if parsed["mode"] == "GraphRAG":
        answer = result.get("answer")
        evidence = _graph_evidence(result, root)
        generator = result.get("cloud_model") or "(云端模型)"
    else:
        generation = result.get("generation") or {}
        if not isinstance(generation, dict):
            raise IntegrationError("Vector RAG 的 generation 字段异常。")
        answer = generation.get("answer")
        evidence = _vector_evidence(result, run.stdout, root)
        generator = generation.get("model") or "(云端模型)"

    if not isinstance(answer, str) or not answer.strip():
        raise IntegrationError("结果 JSON 没有实际生成的云端回答。")

    valid_ids = {item.get("id") for item in evidence}
    cited = set(CITATION_RE.findall(answer))
    missing = sorted(x for x in cited if x not in valid_ids)
    sent = sum(
        item.get("sent_to_model") is True
        for item in evidence if item.get("kind") == "F"
    )
    return {
        "answer": answer,
        "evidence": evidence,
        "meta": {
            "mode": parsed["mode"],
            "generator": _str(generator, 100),
            "vision_requested": parsed["with_images"],
            "vision_images_sent": sent,
            "result_json": path.relative_to(root).as_posix(),
            "graph_facts": len(result.get("graph_facts") or []) if parsed["mode"] == "GraphRAG" else 0,
            "unsupported_citations": missing,
            "retrieval_limit_note": (
                "GraphRAG 使用现有脚本固定文本/图表召回设置；top_k 参数仅用于 Vector RAG。"
                if parsed["mode"] == "GraphRAG" else ""
            ),
        },
    }


def main() -> int:
    try:
        incoming = json.load(sys.stdin)
        output = query_existing_project(incoming)
        # stdout 必须只有一条 JSON 结构，不混入真实脚本的诊断信息。
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except (IntegrationError, json.JSONDecodeError, OSError) as exc:
        print(f"YOLO 前端桥接失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
