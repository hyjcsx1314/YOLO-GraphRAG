# YOLO-GraphRAG

基于四篇 YOLO 英文论文的课程研究项目，涵盖 **4-bit QLoRA 本地微调**、**Milvus 多模态向量检索**、**Neo4j 知识图谱** 和 **Streamlit 可视化问答**。

## 项目介绍

语料为 YOLOX、YOLOv7、YOLOv9 和 YOLOv10 论文。系统有两条彼此独立的工作链路：

- **本地微调实验**：使用 Qwen2.5-1.5B-Instruct，采用 4-bit QLoRA 训练领域 Adapter，使用冻结题集评价模型表现。
- **检索增强问答**：使用 E5、CLIP、Milvus 和 Neo4j 检索论文的文本、图表及实体关系，由配置的**云端多模态模型**生成回答；引用分为 `[G]` 图谱、`[T]` 文本、`[F]` 图表。

本地 QLoRA **不是**在线 GraphRAG 的回答生成器。

## 功能

- Vector RAG / GraphRAG 模式切换及真实后端调用
- 论文图表检索、视觉输入与引用展示
- Neo4j 图谱的只读关系网络可视化
- 本地训练与人工评测结果的可视化展示
- 实验脚本、接口测试与 Cypher 查询

## 目录

```text
frontend/                   Streamlit 页面、CLI 桥接器和 Neo4j 可视化
scripts/                    论文解析、微调、向量检索、图谱、评测
tests/                      核心模块的离线测试
cypher/                     Neo4j 查询
yolo_qlora_v2/              QLoRA V2 训练与评测脚本
yolo_qlora_eval_stage/      冻结测试与人工审核辅助脚本
yolo_alpaca_seed/           指令种子与切分脚本
YOLO_eval_v2_人工构建/       训练和测试数据审阅辅助脚本
docs/results/               两组人工评测汇总
requirements_*.txt          分模块依赖
```

## 主要实验结果

以下数值为原有课程项目的**历史实测摘要**，不是每次启动网页时重新运行的测试。

| 实验 | 结果 |
|---|---|
| QLoRA 冻结测试（37 题）| Base 64.9%；QLoRA V2 82.4% |
| Vector RAG / GraphRAG（12 题）| 4.750 / 6 与 4.833 / 6 |
| 无图 / 有图视觉回答（8 题）| 4.500 / 6 与 5.625 / 6 |
| Neo4j 作者关系补建后 | 83 个节点、108 条关系 |
| 图表检索开发诊断（10 题）| 融合+论文路由 Hit@5 为 100% |

这些小规模实验不支持对未知论文的泛化表现作普遍断言。

## 环境与运行

建议 Python 3.10、WSL2 Ubuntu 22.04，并配置可访问的 Milvus 和 Neo4j AuraDB。

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
python -m pip install -r requirements_ui.txt
python -m pip install -r requirements_multimodal_rag.txt
python -m pip install -r requirements_graphrag_stage1.txt
```

将 `.env.example` 中的**变量名称**用于你本机的环境配置。密钥只在本机设置，不要将真正的 `.env` 提交到仓库。至少需要：

- `RAG_LLM_ENDPOINT`、`RAG_LLM_MODEL`、`RAG_LLM_KEY`
- `NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`

本项目使用既有的 Milvus Collection 和 Neo4j 图谱；首次完整运行需要准备四篇原始论文、对应图表及本地数据资产，并按项目脚本构建索引。**本 GitHub 展示版不包含这些原始数据、Docker 运行卷及训练权重，因此不能仅凭源码直接复现全部实测结果。**

在数据和服务已经配置完成的环境中，可使用：

```bash
python scripts/yolo_multimodal_rag.py check
python scripts/yolo_graph_rag.py check
python -m streamlit run frontend/app.py --server.port 8501
```

浏览器访问：`http://localhost:8501`。云端问答会产生外部 API 请求与相应费用。

## 数据与模型说明

论文原始文件应从作者或 arXiv 等合法来源获取：
[YOLOX](https://arxiv.org/abs/2107.08430)、
[YOLOv7](https://arxiv.org/abs/2207.02696)、
[YOLOv9](https://arxiv.org/abs/2402.13616)、
[YOLOv10](https://arxiv.org/abs/2405.14458)。

源码仓库未附带课程报告（含个人信息）、原始 PDF、提取的论文全文、图像资产、模型权重、账户配置或原始日志。发布这些材料前需分别确认许可及隐私风险。本仓库暂未附加开源许可证。
