#!/usr/bin/env python3
"""RTX 4060/8GB: Qwen2.5-1.5B-Instruct NF4 + LoRA V2 正式微调。

只用 train_v2 153条，不读取验证或测试答案；V1 Adapter 不被覆盖。
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt_format import make_messages

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
os.environ["TOKENIZERS_PARALLELISM"]="false"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(root, package):
    data=root/"data"/"datasets"
    new_train=data/"qlora_train_v2_153.jsonl"
    provenance=package/"qlora_train_v2_153_with_sources.jsonl"
    manifest_path=package/"train_v2_manifest.json"
    freeze_path=data/"eval_frozen_v2"/"freeze_record_v2.json"
    source_path=data/"train_chunks_v3.jsonl"

    for file in (new_train,provenance,manifest_path,freeze_path,source_path):
        if not file.is_file():
            raise SystemExit(f"缺少文件: {file}")

    info=json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(new_train)!=info["training_sha256"]:
        raise SystemExit("V2训练JSONL指纹不匹配，请从审核ZIP重新复制，切勿直接修改。")
    if sha(source_path)!=info["source_sha256"]:
        raise SystemExit("train_chunks_v3.jsonl 与审核时版本不一致，请核对原始来源。")

    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    for rel,wanted in freeze["files_sha256"].items():
        file=data/rel
        if not file.is_file() or sha(file)!=wanted:
            raise SystemExit(f"冻结数据、原V1训练数据或划分清单已被修改: {file}")

    detailed=read_jsonl(provenance)
    rows=read_jsonl(new_train)
    sources={r["chunk_id"]:r for r in read_jsonl(source_path)}
    if len(rows)!=153 or len(detailed)!=153:
        raise SystemExit("样本总数不是已审核的153条。")
    for r,detail in zip(rows,detailed):
        if r!={k:detail[k] for k in ("instruction","input","output")}:
            raise SystemExit("训练样本与审订溯源版不一致。")
        source=sources.get(detail["chunk_id"])
        if not source:
            raise SystemExit(f"训练样本来源不在训练块内: {detail['chunk_id']}")
        if (source["paper_id"],int(source["page"]))!=(detail["paper_id"],int(detail["page"])):
            raise SystemExit("来源论文或PDF页码不一致")
    train_pages={(r["paper_id"],int(r["page"])) for r in detailed}
    frozen_pages={
        (r["paper_id"],int(r["page"]))
        for s in ("val","test")
        for r in freeze["pages_by_split"][s]
    }
    if train_pages & frozen_pages:
        raise SystemExit(f"发现训练页与冻结验证/测试页面重叠: {train_pages & frozen_pages}")

    # 不读取、展示或训练任何验证集/测试集的答案。
    print(f"来源检查通过：153个训练样本，训练PDF页数{len(train_pages)}；未触及冻结验证/测试答案。")
    return rows, new_train, freeze_path


def encode(tokenizer,row,max_length):
    p=make_messages(row["instruction"],row["input"])
    q=make_messages(row["instruction"],row["input"],row["output"])
    prompt_ids=tokenizer.apply_chat_template(
        p, tokenize=True,add_generation_prompt=True
    )
    total_ids=tokenizer.apply_chat_template(
        q, tokenize=True,add_generation_prompt=False
    )
    if total_ids[:len(prompt_ids)]!=prompt_ids:
        raise RuntimeError("Qwen对话模板前缀不一致；已中止，防止错误训练标签。")
    if len(total_ids)>max_length:
        raise ValueError(
            f"样本长度{len(total_ids)}超过{max_length} tokens；"
            "不应截断标准答案，请调整max-length并重做dry-run。"
        )
    if len(total_ids)<=len(prompt_ids):
        raise ValueError("没有Assistant监督输出")
    x=torch.tensor([total_ids],dtype=torch.long)
    y=x.clone()
    y[:, :len(prompt_ids)]=-100
    return x,y,len(total_ids)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--max-length",type=int,default=768)
    ap.add_argument("--grad-acc",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    package=Path(__file__).resolve().parent
    output=root/"models"/"qlora_yolo_v2"
    result_path=root/"results"/"qlora_yolo_v2_metrics.json"
    if min(args.epochs,args.max_length,args.grad_acc,args.lr)<=0:
        raise SystemExit("epochs/max-length/grad-acc/lr 必须为正值")
    if output.exists():
        raise SystemExit("models/qlora_yolo_v2 已存在，禁止覆盖适配器。")

    rows,train_file,freeze_path=preflight(root,package)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    examples=[encode(tokenizer,r,args.max_length) for r in rows]
    max_tokens=max(x[2] for x in examples)
    print(f"完整数据加载成功：{len(rows)} 条，最大token长度={max_tokens}。")
    if args.dry_run:
        print("DRY RUN 通过：仅完成哈希、页面隔离和token长度检查；未加载模型或更新任何参数。")
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA不可用：请检查WSL2 / NVIDIA驱动，不能转为云端训练。")

    rng=random.Random(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quant=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )
    base=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,quantization_config=quant,device_map={"":0},
        dtype=dtype,low_cpu_mem_usage=True
    )
    assert base.is_loaded_in_4bit
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(
        base,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}
    )
    lora=LoraConfig(
        r=8,lora_alpha=16,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],
        bias="none",task_type="CAUSAL_LM"
    )
    model=get_peft_model(base,lora)
    model.print_trainable_parameters()
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=0.01)
    model.train()

    optimizer_steps=0
    loss_by_epoch=[]
    step_history=[]
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()

    for epoch in range(args.epochs):
        indices=list(range(len(examples)))
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.0

        for n,index in enumerate(indices):
            x,y,_=examples[index]
            ids=x.to("cuda")
            labels=y.to("cuda")
            attention=torch.ones_like(ids)
            outputs=model(input_ids=ids,attention_mask=attention,labels=labels)
            loss=outputs.loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Epoch{epoch+1} sample{index}: loss非有限值")
            value=float(loss.item())
            loss_sum+=value
            current_group_start=(n//args.grad_acc)*args.grad_acc
            current_group_size=min(args.grad_acc,len(examples)-current_group_start)
            (loss/current_group_size).backward()
            step_history.append({
                "epoch":epoch+1,"micro_step":n+1,"sample_index":index,"loss":value
            })
            if (n+1)%args.grad_acc==0 or n+1==len(examples):
                torch.nn.utils.clip_grad_norm_(trainable,1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps+=1
            del x,y,ids,labels,attention,outputs,loss

        avg=loss_sum/len(examples)
        loss_by_epoch.append(avg)
        print(f"Epoch {epoch+1}/{args.epochs} | avg train loss={avg:.6f} | optimizer updates={optimizer_steps}",flush=True)

    output.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.cuda.synchronize()

    result_path.parent.mkdir(exist_ok=True,parents=True)
    metrics={
        "model":MODEL_ID,
        "adapter_directory":str(output),
        "local_gpu":torch.cuda.get_device_name(0),
        "training_sha256":sha(train_file),
        "frozen_eval_v2_sha256":sha(freeze_path),
        "dataset_size":len(rows),
        "quantization":"4-bit NF4 with double quantization",
        "lora_r":8,"lora_alpha":16,"lora_dropout":0.05,
        "target_modules":["q_proj","v_proj"],
        "learning_rate":args.lr,
        "epochs":args.epochs,
        "gradient_accumulation":args.grad_acc,
        "max_length":args.max_length,
        "max_observed_tokens":max_tokens,
        "optimizer_steps":optimizer_steps,
        "epoch_average_losses":loss_by_epoch,
        "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
        "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
        "elapsed_seconds":round(time.perf_counter()-started,2)
    }
    result_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    steps_path=result_path.parent/"qlora_yolo_v2_step_losses.jsonl"
    steps_path.write_text(
        "".join(json.dumps(x,ensure_ascii=False)+"\n" for x in step_history),encoding="utf-8"
    )
    print("QLoRA V2已训练完成：",output)
    print("训练指标：",result_path)
    print("逐步Loss：",steps_path)
    print("峰值保留显存MiB：",metrics["peak_reserved_mib"])
    print("训练Loss的变化不代表验证集必然改善；请继续运行冻结的V2验证题。")


if __name__=="__main__":
    main()
