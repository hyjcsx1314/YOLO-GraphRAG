"""确保原模型评测与 QLoRA 训练采用相同的提示词格式。"""

SYSTEM_PROMPT = (
    "你是目标检测领域的学术论文助手。"
    "请严格依据用户提供的英文论文片段回答；"
    "原文缺乏依据时必须说明“原文未说明”。"
    "如果题目要求 JSON，只输出合法 JSON 对象，不能包含 Markdown 代码围栏。"
)

def make_messages(instruction, excerpt, output=None):
    result=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"{instruction}\n\n英文论文片段：\n{excerpt}"}
    ]
    if output is not None:
        result.append({"role":"assistant","content":output})
    return result
