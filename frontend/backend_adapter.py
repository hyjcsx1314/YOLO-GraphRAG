"""Streamlit 至真实 YOLO CLI 桥接器：自动发现 bridge_local.py。

只做进程启动、JSON 校验与图像路径安全检查，不负责重新训练或建库。
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


class BridgeNotConfigured(RuntimeError):
    pass


class BridgeError(RuntimeError):
    pass


def _command() -> list[str] | None:
    override = os.getenv("YOLO_UI_BRIDGE_COMMAND", "").strip()
    if override:
        argv = shlex.split(override)
        return argv or None
    local = Path(__file__).resolve().parent / "bridge_local.py"
    return [sys.executable, str(local)] if local.is_file() else None


def is_configured() -> bool:
    return _command() is not None


def status_label() -> str:
    if is_configured():
        return "已找到真实后端桥接程序（需在本机联网及配置 API / 数据库）"
    return "尚未检测到 bridge_local.py；可继续查看历史实验结果"


def missing_cloud_settings() -> list[str]:
    return [
        key for key in ("RAG_LLM_ENDPOINT", "RAG_LLM_MODEL", "RAG_LLM_KEY")
        if not os.getenv(key, "").strip()
    ]


def missing_graph_settings() -> list[str]:
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    return [key for key in required if not os.getenv(key, "").strip()]


def query_backend(
    project_root: Path,
    question: str,
    mode: str,
    with_images: bool,
    top_k: int,
) -> dict[str, Any]:
    argv = _command()
    if not argv:
        raise BridgeNotConfigured(
            "找不到 frontend/bridge_local.py；请解压联调版前端。"
        )
    missing = missing_cloud_settings()
    if mode == "GraphRAG":
        missing += missing_graph_settings()
    if missing:
        raise BridgeNotConfigured(
            "当前 Streamlit 进程缺少环境变量：" + "、".join(missing)
            + "。请在启动网页前设置这些变量，然后重启 Streamlit。"
        )

    payload = {
        "question": question, "mode": mode,
        "with_images": bool(with_images), "top_k": int(top_k),
    }
    try:
        result = subprocess.run(
            argv,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(project_root),
            timeout=int(os.getenv("YOLO_UI_TIMEOUT", "900")),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BridgeError("后端超时；请检查 Milvus、Neo4j 与云端模型响应。") from exc
    except OSError as exc:
        raise BridgeError("启动后端桥接程序失败，请检查 Python 环境与文件路径。") from exc

    if result.returncode != 0:
        # 仅在服务端终端记录错误；不将可能含敏感信息的 stderr 显示在网页。
        print(
            "[YOLO UI] Bridge exited nonzero. Check script diagnostics below.",
            file=sys.stderr,
        )
        print("\n".join(result.stderr.splitlines()[-8:])[:3500], file=sys.stderr)
        raise BridgeError(
            f"后端桥接程序退出码 {result.returncode}。"
            "请在启动 Streamlit 的 WSL2 终端查看错误原因。"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BridgeError(
            "后端输出不是单个 JSON 对象；日志应写入 stderr。"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise BridgeError("后端响应缺少 answer 字段。")
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        raise BridgeError("后端的 evidence 必须是数组。")
    data["evidence"] = [e for e in evidence if isinstance(e, dict)]
    if not isinstance(data.get("meta"), dict):
        data["meta"] = {}
    return data


def safe_image_path(project_root: Path, supplied: Any) -> Path | None:
    """只显示项目内的图像，禁止目录穿越和任意文件读取。"""
    if not isinstance(supplied, str) or not supplied.strip():
        return None
    root = project_root.resolve()
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to((root / "data").resolve()):
        return None
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    return candidate if candidate.is_file() else None
