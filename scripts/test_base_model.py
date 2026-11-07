import json
from pathlib import Path

import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

assert torch.cuda.is_available(), "CUDA不可用"

# RTX 4060支持BF16时优先使用
compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print("正在加载模型:", MODEL_ID)
print("计算精度:", compute_dtype)

# 4-bit NF4 + 双重量化
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=compute_dtype,
    low_cpu_mem_usage=True,
)

model.eval()

# 确认真正进行了4-bit量化
num_4bit = sum(
    isinstance(m, bnb.nn.Linear4bit)
    for m in model.modules()
)

print("模型加载成功!")
print("4-bit状态:", getattr(model, "is_loaded_in_4bit", False))
print("4-bit线性层数量:", num_4bit)

# 原模型测试问题
questions = [
    {
        "task": "术语解释",
        "prompt": (
            "请用中文解释YOLO目标检测中的"
            "Precision、Recall和mAP@0.5，"
            "并说明它们的区别。"
        ),
    },
    {
        "task": "结构化实体抽取",
        "prompt": (
            "从下面的描述中提取method、dataset、"
            "metric三个字段，只输出JSON："
            "YOLOv7是一种目标检测方法，"
            "在MS COCO数据集上使用AP指标进行评估。"
        ),
    },
]

records = []

for item in questions:
    messages = [
        {
            "role": "system",
            "content": "你是一个目标检测领域的学术助手。",
        },
        {
            "role": "user",
            "content": item["prompt"],
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    )

    print("\n======================")
    print("任务:", item["task"])
    print("问题:", item["prompt"])
    print("原模型回答:\n", answer)

    records.append({
        "model": MODEL_ID,
        "task": item["task"],
        "prompt": item["prompt"],
        "answer": answer,
    })

# 保存原模型测试记录
output_path = RESULT_DIR / "baseline_qwen1_5b.json"
output_path.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n测试结果已保存:", output_path)
print(
    "当前PyTorch显存分配(MiB):",
    round(torch.cuda.memory_allocated() / 1024**2),
)
print(
    "PyTorch显存峰值保留(MiB):",
    round(torch.cuda.max_memory_reserved() / 1024**2),
)
