"""YOLO-GraphRAG 课程项目 Streamlit 前端。

显示项目实测指标，并通过标准化桥接器调用原有问答程序。
启动：python -m streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend_adapter import (
    BridgeError,
    BridgeNotConfigured,
    is_configured as ask_configured,
    missing_cloud_settings,
    missing_graph_settings,
    query_backend,
    safe_image_path,
    status_label,
)
from neo4j_view import configured as neo4j_configured, read_live_graph


ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="YOLO · GraphRAG | 学术论文智能问答",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {--ink:#16283D; --teal:#087E85; --muted:#63758A;}
    [data-testid="stAppViewContainer"] {background:#F7F9FC;}
    .block-container {max-width:1320px;padding-top:2.1rem;padding-bottom:3rem;}
    [data-testid="stSidebar"] {background:#EEF3F8;}
    h1,h2,h3 {letter-spacing:-.018em;}
    .hero {
       background:linear-gradient(115deg,#153F51 0%,#125F65 57%,#158C83 100%);
       color:#fff;border-radius:19px;padding:31px 35px;margin-bottom:23px;
       box-shadow:0 12px 32px rgba(18,63,81,.12);
    }
    .hero .eyebrow {font-size:.76rem;letter-spacing:.18em;opacity:.78;
                    margin-bottom:9px;font-weight:700;}
    .hero .headline {font-size:2.03rem;line-height:1.3;font-weight:750;}
    .hero .subline {font-size:.99rem;line-height:1.85;opacity:.9;margin-top:9px;}
    .info-panel {background:#fff;border:1px solid #E3EAF2;padding:17px 21px;
                 border-radius:12px;margin-bottom:10px;color:#24364A;}
    .info-panel strong {color:#116D76;}
    .mini-label {color:#687B8F;font-size:.82rem;letter-spacing:.025em;}
    div[data-testid="stMetric"] {background:white;border:1px solid #E1E8F0;
       border-radius:13px;padding:16px 18px;min-height:117px;}
    div[data-testid="stMetricValue"] {font-weight:750;}
    </style>
    """,
    unsafe_allow_html=True,
)


def header(eyebrow: str, headline: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="headline">{headline}</div>'
        f'<div class="subline">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def panel(text: str) -> None:
    # 仅用于程序内静态文本，不接收用户问题或模型回答。
    st.markdown(f'<div class="info-panel">{text}</div>', unsafe_allow_html=True)


def show_evidence(evidences: list[dict]) -> None:
    if not evidences:
        return
    with st.expander(f"查看引用与多模态证据（{len(evidences)} 条）", expanded=False):
        for idx, item in enumerate(evidences, start=1):
            kind = str(item.get("kind", "?")).upper()
            citation = str(item.get("id", f"{kind}{idx}"))
            title = str(item.get("title", f"证据 {idx}"))
            st.markdown(f"**[{citation}] {title}**")
            snippet = item.get("snippet", "")
            if snippet:
                st.text(str(snippet))
            local_image = safe_image_path(ROOT, item.get("image_path"))
            if local_image:
                st.image(str(local_image), caption=local_image.name, use_container_width=True)
                if item.get("sent_to_model"):
                    st.caption("此图在本次视觉模式中实际发送给云端模型。")
                elif kind == "F":
                    st.caption("这是检索到的图表，本次未作为图像像素发送给云端模型。")
            elif kind == "F" and item.get("asset_id"):
                st.caption(f"本地图像未找到：{item['asset_id']}（检索记录仍有效）")
            st.divider()


def chat_page() -> None:
    header(
        "ACADEMIC ASSISTANT  /  YOLO SERIES",
        "YOLO 学术论文智能问答",
        "在已构建的 Milvus 文本 / 图表证据与 Neo4j 图谱上进行问答，"
        "保留 [G]、[T]、[F] 来源标记。",
    )
    left, middle, right = st.columns([1.15, 1, 1])
    with left:
        mode = st.radio(
            "问答路径",
            ["GraphRAG", "Vector RAG"],
            horizontal=True,
            index=0,
            help="GraphRAG：图谱扩展 + 向量证据；Vector RAG：向量证据",
        )
    with middle:
        with_images = st.toggle(
            "向多模态模型提供图像输入",
            value=False,
            help="这并非重新训练 CLIP；仅控制调用端是否发送召回的图像",
        )
    with right:
        top_k = st.slider(
            "检索条数（仅 Vector RAG）",
            1, 10, 5,
            disabled=(mode == "GraphRAG"),
            help="GraphRAG 的原有脚本固定文本 6 / 图表 5；此处不改变已完成实验设置。",
        )

    if not ask_configured():
        st.warning("未检测到 frontend/bridge_local.py；请安装联调版。")
    else:
        missing = missing_cloud_settings()
        if mode == "GraphRAG":
            missing += missing_graph_settings()
        if missing:
            st.warning(
                "当前 Streamlit 进程尚未配置：" + "、".join(missing)
                + "。请在启动网页前设置变量，然后重新启动 Streamlit。"
            )
        else:
            st.success(
                "已具备后端调用所需的环境变量；实际 Milvus / Neo4j / 云端 API 连通性"
                "以点击提问后的结果为准。"
            )
    st.caption(
        "选择视觉模式将把检索到的部分论文图表发送给已配置的云端模型，"
        "可能消耗 API 额度；这不是本地 QLoRA 推理。"
    )

    st.markdown("**试试这些问题**")
    prompts = [
        "YOLOv9 的 PGI 机制如何改善梯度信息传递？",
        "YOLOv10 是如何实现 NMS-free 检测的？",
        "比较 YOLOX 与 YOLOv7 的主要结构改进。",
    ]
    boxes = st.columns(3)
    selected = None
    for i, prompt in enumerate(prompts):
        with boxes[i]:
            if st.button(prompt, key=f"sample_{i}", use_container_width=True):
                selected = prompt

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message["role"] == "assistant":
                st.caption(message.get("meta", ""))
                if message.get("unsupported_citations"):
                    st.warning(
                        "检测到回答使用了未匹配证据的引用编号："
                        + "、".join(message["unsupported_citations"])
                        + "。请核对原脚本输出及论文来源。"
                    )
                show_evidence(message.get("evidence", []))
                if message.get("result_json"):
                    st.caption("真实后端记录：" + str(message["result_json"]))

    typed = st.chat_input("输入关于 YOLOX、YOLOv7、YOLOv9、YOLOv10 的学术问题……")
    question = typed or selected
    if question and question.strip():
        st.session_state.chat_history.append({"role": "user", "text": question})
        t0 = time.perf_counter()
        try:
            with st.spinner("正在调用现有后端并检索论文证据……"):
                reply = query_backend(ROOT, question, mode, with_images, top_k)
            elapsed = round(time.perf_counter() - t0, 2)
            record_meta = reply.get("meta", {})
            sent = record_meta.get("vision_images_sent", 0)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "text": reply["answer"],
                    "evidence": reply.get("evidence", []),
                    "meta": (
                        f"{mode} · 云端模型：{record_meta.get('generator', '未报告')}"
                        f" · 实际发送图片：{sent} 张 · 端到端耗时 {elapsed} s"
                    ),
                    "unsupported_citations": record_meta.get("unsupported_citations", []),
                    "result_json": record_meta.get("result_json", ""),
                }
            )
        except (BridgeNotConfigured, BridgeError) as err:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "text": f"⚠️ **本次调用未成功：** {err}",
                    "evidence": [],
                    "meta": "此消息是前端状态提示，并非模型生成结果。",
                }
            )
        st.rerun()

    if st.session_state.chat_history:
        if st.button("清空当前对话", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


@st.cache_data(show_spinner=False, ttl=120)
def list_local_images(project_root: str) -> list[str]:
    base = Path(project_root) / "data"
    if not base.exists():
        return []
    found: list[str] = []
    excluded = {"volumes", "cache", ".git", "__pycache__", ".venv", ".venv-rag"}
    for current, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith(".")]
        for name in names:
            if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            found.append(str(Path(current) / name))
            if len(found) >= 1500:
                return found
    return sorted(
        found,
        key=lambda p: (0 if "visual_overrides" in p else 1, p.lower()),
    )


def images_page() -> None:
    header(
        "MULTIMODAL EVIDENCE",
        "论文图表与文本证据",
        "展示项目 data/ 中的本地图表；实时检索所得图像将在「智能问答」中跟随引用显示。",
    )
    col1, col2, col3, col4 = st.columns(4)
    for col, label, count in zip(
        [col1, col2, col3, col4],
        ["Text (E5)", "Figures (CLIP)", "Visual Meta (E5)", "PDF Pages (E5)"],
        [78, 20, 20, 231],
    ):
        col.metric(label, count)
    st.caption("以上是 2026-11-05 实验记录中的数据库规模快照，不代表每次网页打开都重新查询 Milvus。")
    imgs = list_local_images(str(ROOT))
    if not imgs:
        st.warning(
            "当前工作目录未找到 data/ 内的 PNG/JPG/WebP。"
            "请将 frontend/ 放到原项目根目录，或检查图表实际存储路径。"
        )
        return

    papers = ["全部论文", "YOLOX", "YOLOv7", "YOLOv9", "YOLOv10"]
    chosen = st.selectbox("按论文筛选", papers)
    filtered = [
        p for p in imgs
        if chosen == "全部论文" or chosen.lower() in Path(p).name.lower()
    ]
    if not filtered:
        st.info("当前筛选下没有找到本地图像。")
        return
    st.caption(f"已找到 {len(filtered)} 张本地图像；图库不等同于 Milvus 向量检索结果。")
    image_choice = st.selectbox(
        "选择论文图表",
        filtered,
        format_func=lambda x: str(Path(x).relative_to(ROOT)),
    )
    st.image(image_choice, caption=str(Path(image_choice).relative_to(ROOT)),
             use_container_width=True)
    if "YOLOv9_F3" in image_choice:
        st.info(
            "如选择了 data/visual_overrides/YOLOv9_F3.png，"
            "这是图像输入覆盖修复文件；不能将它描述为重新训练 CLIP。"
        )


def plot_neo4j_edges(edges: list[dict]) -> go.Figure:
    graph = nx.DiGraph()
    for e in edges:
        a, b = str(e["source_id"]), str(e["target_id"])
        graph.add_node(a, name=str(e["source"]), kind=str(e["source_type"]))
        graph.add_node(b, name=str(e["target"]), kind=str(e["target_type"]))
        graph.add_edge(a, b, relation=str(e["relation"]))
    if not graph.nodes:
        return go.Figure()

    pos = nx.spring_layout(graph, seed=11, iterations=90)
    edge_x, edge_y = [], []
    for s, t in graph.edges():
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]

    node_x, node_y, titles, labels = [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        full_name = data.get("name", node)
        labels.append(full_name[:18] + ("…" if len(full_name) > 18 else ""))
        titles.append(f"{full_name}<br>Type: {data.get('kind', 'Unknown')}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.1, color="#BBCBD9"), hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=labels,
            textposition="top center", textfont=dict(size=10, color="#263D50"),
            marker=dict(size=14, color="#167C87", line=dict(width=1, color="#FFFFFF")),
            hovertext=titles, hoverinfo="text", showlegend=False,
        )
    )
    fig.update_layout(
        height=590, margin=dict(l=10, r=10, t=26, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title=dict(text="Neo4j Relationship Network", font=dict(size=16)),
    )
    return fig


def graph_page() -> None:
    header(
        "KNOWLEDGE GRAPH / NEO4J",
        "YOLO 论文知识图谱",
        "展示关系结构、实体类型与一至两跳检索的知识基础；"
        "图谱查询采用只读连接，不触发重建或写库。",
    )
    cards = st.columns(4)
    for col, label, number in zip(
        cards,
        ["Nodes", "Relationships", "Entity Types", "Relation Types"],
        [83, 108, 9, 7],
    ):
        col.metric(label, number)
    st.caption("上方指标来自 2026-11-05 的实验记录。")

    panel(
        "<strong>主要关系：</strong> AUTHORED_BY · PROPOSES · USES · "
        "EVALUATED_ON · HAS_COMPONENT · ILLUSTRATED_BY · REPORTS_METRIC"
    )
    if not neo4j_configured():
        st.info(
            "未检测到 Neo4j 连接环境变量。可设置 NEO4J_URI、"
            "NEO4J_USERNAME（或 NEO4J_USER）、NEO4J_PASSWORD；"
            "图谱统计快照仍可浏览。"
        )
    count = st.slider("最多读取关系边数", 10, 150, 70, 10)
    if st.button("只读连接并获取当前 Neo4j 图谱", type="primary"):
        try:
            with st.spinner("正在读取当前图谱……"):
                st.session_state.live_graph = read_live_graph(count)
        except Exception as exc:
            st.error(f"Neo4j 读取失败：{type(exc).__name__}。请检查环境变量、网络与数据库权限。")

    live = st.session_state.get("live_graph")
    if live:
        st.success(
            f"当前数据库读取结果：{live['nodes']} 个节点，"
            f"{live['relationships']} 条关系（可能与历史快照不同）。"
        )
        st.plotly_chart(plot_neo4j_edges(live["edges"]), use_container_width=True)
        x, y = st.columns(2)
        with x:
            st.markdown("**Node Types**")
            st.dataframe(pd.DataFrame(live["node_types"]), hide_index=True,
                         use_container_width=True)
        with y:
            st.markdown("**Relationship Types**")
            st.dataframe(pd.DataFrame(live["rel_types"]), hide_index=True,
                         use_container_width=True)
        st.caption("网页最多展示所选数量的关系边，并非完整图谱。")
    else:
        st.markdown("**历史实测节点类型分布**")
        node_df = pd.DataFrame(
            [
                ("Author", 16), ("Component", 7), ("Dataset", 3),
                ("Figure", 9), ("Method", 26), ("Metric", 1),
                ("Model", 14), ("Paper", 4), ("Table", 3),
            ],
            columns=["Type", "Count"],
        ).set_index("Type")
        st.bar_chart(node_df, y="Count")
        st.caption("尚未读取当前 Neo4j 实时关系图；请点击上方按钮。")


def evaluation_page() -> None:
    header(
        "EVALUATION / VERIFIED SNAPSHOT",
        "模型性能与实验结果",
        "仅使用进度交接中已人工评分或已实测的数据；开发诊断与独立盲测严格区分。",
    )
    st.subheader("01 · QLoRA 本地微调")
    a, b = st.columns([1.3, 1])
    with a:
        score = pd.DataFrame(
            {
                "Model": ["Base", "QLoRA V1", "QLoRA V2"],
                "Accuracy (%)": [64.9, 56.8, 82.4],
            }
        ).set_index("Model")
        st.bar_chart(score, y="Accuracy (%)")
    with b:
        st.metric("QLoRA V2 Frozen Test", "82.4%", "17.5 pp vs Base")
        st.markdown(
            "**37 题冻结测试**：Base 24/37；V1 21/37；V2 30.5/37。"
        )
        st.markdown("**严格 JSON 合规（6 题）**：Base 0/6；V1 2/6；V2 6/6。")
        st.caption("V2 与 V1 同时改变了样本量及训练步数，不能归因于单一参数。")

    st.divider()
    st.subheader("02 · Vector RAG vs GraphRAG")
    c, d = st.columns([1.2, 1])
    with c:
        data = pd.DataFrame(
            {"Approach": ["Vector RAG", "GraphRAG"],
             "Mean Score / 6": [4.750, 4.833]}
        ).set_index("Approach")
        st.bar_chart(data, y="Mean Score / 6")
    with d:
        st.metric("GraphRAG Mean Score", "4.833 / 6", "+0.083")
        st.markdown("12 道固定题、24 条回答、关闭视觉；11 题同分，1 题 GraphRAG 高 1 分。")
        st.caption("差异很小，不能据此声称 GraphRAG 普遍优于 Vector RAG。")

    st.divider()
    st.subheader("03 · Vision A/B")
    e, f = st.columns([1.2, 1])
    with e:
        vision = pd.DataFrame(
            {"Mode": ["NoImage", "WithImage"],
             "Mean Score / 6": [4.500, 5.625]}
        ).set_index("Mode")
        st.bar_chart(vision, y="Mean Score / 6")
    with f:
        st.metric("WithImage Mean Score", "5.625 / 6", "+1.125")
        st.markdown("8/8 完整配对；7 题提升，1 题下降。")
        st.caption(
            "改善主要来自图示细节；VAB04 对 YOLOv7 Figure 5 的监督关系存在误述。"
        )

    st.divider()
    st.subheader("04 · Figure Retrieval Dev Diagnostics")
    hit = pd.DataFrame(
        [
            ("CLIP", 20, 50, 50),
            ("E5 Caption", 70, 90, 90),
            ("Global RRF", 80, 90, 90),
            ("RRF + Paper Route", 80, 90, 100),
        ],
        columns=["Method", "Hit@1 (%)", "Hit@3 (%)", "Hit@5 (%)"],
    )
    st.dataframe(hit, hide_index=True, use_container_width=True)
    st.caption("仅 10 道开发诊断题，衡量目标图表检索命中率，不是问答准确率或独立盲测。")

    export = {
        "source": "YOLO_GraphRAG_项目进度交接_2026-11-05.md",
        "qlora_frozen_test": {
            "base": "24/37", "v1": "21/37", "v2": "30.5/37",
        },
        "graphrag_vs_vector": {"vector": 4.750, "graph": 4.833, "n": 12},
        "vision_ab": {"noimage": 4.500, "withimage": 5.625, "n": 8},
        "neo4j": {"nodes": 83, "relationships": 108},
    }
    st.download_button(
        "下载实测指标快照 JSON",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name="yolo_verified_metrics_snapshot.json",
        mime="application/json",
    )


def overview_page() -> None:
    header(
        "SYSTEM ARCHITECTURE / COURSE PROJECT",
        "YOLO · QLoRA · Multimodal GraphRAG",
        "基于四篇 YOLO 研究论文，将本地轻量微调、跨模态向量检索和知识图谱结合。",
    )
    st.subheader("项目架构")
    st.code(
        """YOLOX / YOLOv7 / YOLOv9 / YOLOv10  (4 PDFs · 61 pages)
                 │
       ┌─────────┴───────────┐
       ▼                     ▼
   Text / Figures       Entity Extraction
       │                     │
       ▼                     ▼
  Milvus (4 collections)  Neo4j (83 nodes / 108 relations)
       │                     │
       └──────────┬──────────┘
                  ▼
            Evidence Retrieval
      [T] Text · [F] Figure · [G] Graph
                  │
                  ▼
          Cloud Multimodal LLM
                  │
                  ▼
         Streamlit Web Interface

Separate local experiment:
  Qwen2.5-1.5B-Instruct → 4-bit QLoRA → Frozen Test""",
        language="text",
    )
    st.info(
        "注意：已实测 GraphRAG 问答使用云端模型，不能描述为由本地 QLoRA 直接生成。"
        "网页只是接入与展示层，不改动已经完成的模型、向量库和图谱。"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        panel("<strong>数据层</strong><br>4 篇论文 · 61 PDF 页<br>20 张精选 A 级图表")
    with col2:
        panel("<strong>检索层</strong><br>E5 + CLIP + RRF<br>Milvus + Neo4j")
    with col3:
        panel("<strong>生成层</strong><br>GraphRAG 与 Vector RAG<br>附来源引用的回答")
    st.subheader("本机部署")
    st.code(
        """cd ~/projects/yolo-graphrag
source .venv-rag/bin/activate
python -m pip install -r requirements_ui.txt
python -m streamlit run frontend/app.py --server.port 8501""",
        language="bash",
    )
    st.caption("在 Windows 浏览器打开 http://localhost:8501，前提是 WSL2 端口转发正常。")


with st.sidebar:
    st.markdown("## 🔬 YOLO · GraphRAG")
    st.caption("大数据与人工智能 · 课程项目")
    st.divider()
    page = st.radio(
        "导航",
        [
            "🧠 智能问答",
            "🖼️ 图表证据",
            "🕸️ 知识图谱",
            "📊 实验结果",
            "🏠 系统概览",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**后端状态**")
    st.caption(status_label())
    st.markdown("**Neo4j 环境**")
    st.caption("连接变量已设置" if neo4j_configured() else "连接变量未设置")
    st.divider()
    st.caption("实验记录：2026-11-05")
    st.caption("实时回答使用已配置的云端模型，历史指标为已实测快照。")

if page.startswith("🧠"):
    chat_page()
elif page.startswith("🖼️"):
    images_page()
elif page.startswith("🕸️"):
    graph_page()
elif page.startswith("📊"):
    evaluation_page()
else:
    overview_page()
