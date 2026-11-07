import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path("models/qlora_smoke_adapter")
RESULT_FILE = Path("results/qlora_smoke_metrics.json")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

assert torch.cuda.is_available(), "CUDA不可用"

dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

# 1. 加载4-bit模型
print("第一步：加载4-bit基础模型", flush=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=dtype,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=dtype,
    low_cpu_mem_usage=True,
)

assert model.is_loaded_in_4bit
model.config.use_cache = False

# 2. 准备低显存训练
print("第二步：准备QLoRA训练", flush=True)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.train()
model.print_trainable_parameters()

# 3. 演示训练样本
# 注意：这是流程测试样本，不是正式论文数据集
messages = [
    {
        "role": "system",
        "content": "你是YOLO目标检测论文信息抽取助手。",
    },
    {
        "role": "user",
        "content": (
            "提取方法、数据集、评价指标，只输出JSON："
            "YOLOX在COCO数据集上使用AP进行评估。"
        ),
    },
]

answer = (
    '{"method":"YOLOX",'
    '"dataset":"COCO",'
    '"metric":"AP"}'
)

prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)

full_ids = tokenizer.apply_chat_template(
    messages + [
        {"role": "assistant", "content": answer}
    ],
    tokenize=True,
    add_generation_prompt=False,
)

assert full_ids[:len(prompt_ids)] == prompt_ids
assert len(full_ids) <= 256

input_ids = torch.tensor(
    [full_ids], dtype=torch.long, device="cuda"
)

attention_mask = torch.ones_like(input_ids)

# 只对Assistant回答部分计算训练损失
labels = input_ids.clone()
labels[:, :len(prompt_ids)] = -100

assert (labels != -100).any()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
)

# 记录一个LoRA权重，检查训练后是否发生变化
tracked_param = next(
    p for name, p in model.named_parameters()
    if "lora_B" in name and p.requires_grad
)
before = tracked_param.detach().float().cpu().clone()

# 4. 真正执行反向传播与参数更新
print("第三步：开始3步QLoRA训练", flush=True)

torch.cuda.reset_peak_memory_stats()
losses = []

for step in range(3):
    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    assert torch.isfinite(loss).item(), "Loss不是有限数值"

    loss.backward()
    optimizer.step()

    loss_value = loss.item()
    losses.append(loss_value)

    print(
        f"Step {step+1}/3 | Loss: {loss_value:.6f}",
        flush=True,
    )

    del outputs, loss

torch.cuda.synchronize()

weight_change = (
    tracked_param.detach().float().cpu() - before
).abs().max().item()

print("LoRA权重最大变化量:", weight_change)

assert weight_change > 0, "LoRA权重没有更新"

# 5. 保存适配器和实验指标
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

model.save_pretrained(ADAPTER_DIR)

peak_allocated = round(
    torch.cuda.max_memory_allocated() / 1024**2
)
peak_reserved = round(
    torch.cuda.max_memory_reserved() / 1024**2
)

metrics = {
    "model": MODEL_ID,
    "quantization": "4-bit NF4",
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "training_steps": 3,
    "losses": losses,
    "weight_max_change": weight_change,
    "peak_allocated_mib": peak_allocated,
    "peak_reserved_mib": peak_reserved,
}

RESULT_FILE.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nQLoRA训练测试成功!")
print("适配器目录:", ADAPTER_DIR)
print("实验指标:", RESULT_FILE)
print("训练峰值分配显存(MiB):", peak_allocated)
print("训练峰值保留显存(MiB):", peak_reserved)
