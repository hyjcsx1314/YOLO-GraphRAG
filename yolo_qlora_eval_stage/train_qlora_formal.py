#!/usr/bin/env python3
"""RTX 4060 8GB：对已人工审定的 Alpaca 训练集进行 Qwen2.5-1.5B-Instruct 4-bit QLoRA。

只读取训练集；不把验证集、测试集答案用于反向传播。
"""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt_format import make_messages

os.environ["TOKENIZERS_PARALLELISM"]="false"
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"


def prepare_example(tokenizer, entry, max_length):
    prompt=make_messages(entry["instruction"],entry["input"])
    full=make_messages(entry["instruction"],entry["input"],entry["output"])
    prompt_ids=tokenizer.apply_chat_template(
        prompt,tokenize=True,add_generation_prompt=True
    )
    ids=tokenizer.apply_chat_template(
        full,tokenize=True,add_generation_prompt=False
    )
    if ids[:len(prompt_ids)]!=prompt_ids:
        raise RuntimeError("Chat Template 的提示与完整答案前缀不一致")
    if len(ids)>max_length:
        raise RuntimeError(
            f"样本 token 长度 {len(ids)} 超过 MAX_LENGTH={max_length}。"
            "不要截断标准答案；先检查训练片段长度或谨慎调高最大长度。"
        )
    if len(ids)<=len(prompt_ids):
        raise RuntimeError("没有用于监督训练的答案 token")
    token_ids=torch.tensor([ids],dtype=torch.long)
    labels=token_ids.clone()
    labels[:, :len(prompt_ids)]=-100
    return token_ids,labels,len(ids)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--max-length",type=int,default=768)
    ap.add_argument("--grad-acc",type=int,default=8)
    ap.add_argument("--lr",type=float,default=0.0001)
    ap.add_argument("--dry-run",action="store_true",help="只检查训练token长度，不加载模型或训练")
    args=ap.parse_args()

    root=Path(args.project).expanduser().resolve()
    train_file=root/"data"/"datasets"/"qlora_train_reviewed.jsonl"
    freeze=root/"data"/"datasets"/"eval_frozen_v1"/"freeze_record.json"
    output=root/"models"/"qlora_yolo_v1"
    metrics_file=root/"results"/"qlora_yolo_v1_metrics.json"

    assert torch.cuda.is_available(),"RTX 4060 CUDA 不可用"
    assert train_file.exists(),f"缺少训练集: {train_file}"
    assert freeze.exists(),"请先冻结验证集与测试集，再开始正式训练。"
    # 在训练前检查冻结试题和划分清单的 SHA256 指纹
    freeze_record=json.loads(freeze.read_text(encoding="utf-8"))
    for filename,expected in freeze_record["sha256"].items():
        path_to_check=(
            root/"data"/"datasets"/filename
            if filename=="split_manifest_v3.json"
            else freeze.parent/filename
        )
        if not path_to_check.exists():
            raise SystemExit(f"冻结数据文件缺失: {path_to_check}")
        actual=hashlib.sha256(path_to_check.read_bytes()).hexdigest()
        if actual!=expected:
            raise SystemExit(f"冻结评测数据被修改，训练中止: {path_to_check}")
    if output.exists():
        raise SystemExit("适配器目录已存在，为防止覆盖真实训练结果，请更换目录名或备份后重新训练。")
    if args.epochs<=0 or args.grad_acc<=0 or args.max_length<=0:
        raise SystemExit("训练参数必须为正数")

    rng=random.Random(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    rows=[
        json.loads(s) for s in train_file.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if not rows:
        raise RuntimeError("训练集为空")
    if not all(set(x)=={"instruction","input","output"} for x in rows):
        raise RuntimeError("训练文件必须恰好包含 Alpaca 的 instruction/input/output 三列")

    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    samples=[prepare_example(tokenizer,item,args.max_length) for item in rows]
    max_seen=max(s[2] for s in samples)
    print(f"样本: {len(samples)} | 最大 token 长度: {max_seen} | 精度: {dtype}",flush=True)
    if args.dry_run:
        print("dry-run 成功：数据与长度检查通过，未加载模型、未训练。")
        return

    cfg=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )
    base=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=cfg,
        device_map={"":0},dtype=dtype,low_cpu_mem_usage=True
    )
    assert base.is_loaded_in_4bit
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}
    )
    lora=LoraConfig(
        r=8,lora_alpha=16,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],
        bias="none",task_type="CAUSAL_LM"
    )
    model=get_peft_model(base,lora)
    model.print_trainable_parameters()

    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    losses=[]
    optimizer_steps=0
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for epoch in range(args.epochs):
        ordering=list(range(len(samples)))
        rng.shuffle(ordering)
        opt.zero_grad(set_to_none=True)
        epoch_loss_sum=0.0
        for step,j in enumerate(ordering):
            input_ids, labels, toklen=samples[j]
            ids=input_ids.to("cuda")
            lab=labels.to("cuda")
            attn=torch.ones_like(ids)
            result=model(input_ids=ids,attention_mask=attn,labels=lab)
            loss=result.loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("训练损失出现 NaN/Inf，停止训练")
            actual=float(loss.item())
            epoch_loss_sum+=actual
            # 正确处理最后一个不满梯度累积长度的组
            group_start=(step//args.grad_acc)*args.grad_acc
            group_length=min(args.grad_acc,len(samples)-group_start)
            (loss/group_length).backward()
            if (step+1)%args.grad_acc==0 or step+1==len(samples):
                torch.nn.utils.clip_grad_norm_(params,1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                optimizer_steps+=1
            del result,loss,ids,lab,attn

        epoch_average=epoch_loss_sum/len(samples)
        losses.append(epoch_average)
        print(f"Epoch {epoch+1}/{args.epochs} | avg train loss: {epoch_average:.6f}",flush=True)

    output.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.cuda.synchronize()

    result={
        "model":MODEL_ID,
        "training_file":str(train_file),
        "training_sha256":hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "dataset_size":len(rows),
        "quantization":"NF4 4-bit + double quant",
        "lora_rank":8,"lora_alpha":16,"lora_dropout":0.05,
        "target_modules":["q_proj","v_proj"],
        "epochs":args.epochs,
        "gradient_accumulation":args.grad_acc,
        "max_length":args.max_length,
        "optimizer_steps":optimizer_steps,
        "epoch_average_losses":losses,
        "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
        "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
        "eval_freeze_sha256":hashlib.sha256(freeze.read_bytes()).hexdigest()
    }
    metrics_file.parent.mkdir(parents=True,exist_ok=True)
    metrics_file.write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("\nQLoRA 完成 | 适配器:",output,flush=True)
    print("训练日志:",metrics_file,flush=True)
    print("GPU 峰值保留显存(MiB):",result["peak_reserved_mib"],flush=True)
    print("注意：训练 Loss 下降并不代表泛化性能提升，需要后续独立评测。")


if __name__=="__main__":
    main()
