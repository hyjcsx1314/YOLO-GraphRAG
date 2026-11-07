#!/usr/bin/env python3
"""使用与V1相同的推理提示模板/量化配置评测独立V2 Adapter。
只创建 tuned_v2_val/test.jsonl，不覆盖既有 base_val/tuned_val 结果。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
from prompt_format import make_messages

os.environ["TOKENIZERS_PARALLELISM"]="false"
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"


def normalized(value):
    if isinstance(value,dict):
        return {k:normalized(v) for k,v in sorted(value.items())}
    if isinstance(value,list):
        return sorted(
            [normalized(v) for v in value],
            key=lambda v:json.dumps(v,sort_keys=True,ensure_ascii=False)
        )
    if isinstance(value,str):
        return " ".join(value.split()).casefold()
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project",default=".")
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--adapter",default="models/qlora_yolo_v2")
    args=p.parse_args()

    root=Path(args.project).expanduser().resolve()
    data=root/"data"/"datasets"
    freeze=data/"eval_frozen_v2"/"freeze_record_v2.json"
    rows_file=data/"eval_frozen_v2"/f"eval_{args.split}_locked_v2.jsonl"
    adapter=root/args.adapter

    if not freeze.exists() or not rows_file.exists():
        raise SystemExit("V2冻结数据不完整，请勿运行评测。")
    freeze_info=json.loads(freeze.read_text(encoding="utf-8"))
    for relative,wanted in freeze_info["files_sha256"].items():
        file=data/relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=wanted:
            raise SystemExit(f"冻结内容发生变化，禁止评测: {file}")
    if not (adapter/"adapter_config.json").is_file():
        raise SystemExit(f"未找到V2 LoRA适配器: {adapter}")
    if not torch.cuda.is_available():
        raise SystemExit("RTX 4060 CUDA不可用")

    output_dir=root/"results"/"eval_v2"
    output_dir.mkdir(parents=True,exist_ok=True)
    output_file=output_dir/f"tuned_v2_{args.split}.jsonl"
    if output_file.exists():
        raise SystemExit(f"结果文件已存在，不能覆盖：{output_file}")

    rows=[
        json.loads(line)
        for line in rows_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != (23 if args.split=="val" else 37):
        raise SystemExit("评测题目数与V2审核集不一致")

    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,quantization_config=quant,device_map={"":0},dtype=dtype
    )
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()

    records=[]
    json_total=strict_valid=matched=0
    with torch.inference_mode():
        for index,row in enumerate(rows,1):
            message=make_messages(row["instruction"],row["input"])
            ids=tok.apply_chat_template(
                message,tokenize=True,add_generation_prompt=True,return_tensors="pt"
            ).to("cuda")
            result=model.generate(
                input_ids=ids,attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,pad_token_id=tok.eos_token_id
            )
            prediction=tok.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
            record={
                "review_id":row["review_id"],
                "paper_id":row["paper_id"],
                "page":row["page"],
                "task_type":row["task_type"],
                "instruction":row["instruction"],
                "input":row["input"],
                "gold":row["output"],
                "prediction":prediction,
            }
            if row["task_type"] in ("structured_json","entity_json"):
                json_total+=1
                try:
                    parsed=json.loads(prediction)
                    valid=isinstance(parsed,dict)
                    record["strict_json_valid"]=valid
                    strict_valid+=int(valid)
                    correct=valid and normalized(parsed)==normalized(json.loads(row["output"]))
                    record["json_fields_match"]=correct
                    matched+=int(correct)
                except (ValueError,TypeError):
                    record["strict_json_valid"]=False
                    record["json_fields_match"]=False
            records.append(record)
            print(f"{index}/{len(rows)} | {row['review_id']} | {prediction[:100]!r}",flush=True)

    output_file.write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),
        encoding="utf-8"
    )
    print("\nV2 Adapter验证完成：",output_file)
    print(f"严格JSON: {strict_valid}/{json_total}；字段匹配: {matched}/{json_total}")
    print("学术问答仍须按原模型相同标准人工审核。")


if __name__=="__main__":
    main()
