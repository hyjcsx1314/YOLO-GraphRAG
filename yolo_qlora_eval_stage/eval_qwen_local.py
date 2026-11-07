#!/usr/bin/env python3
"""4-bit 原模型/QLoRA 适配器评测；学术问答输出供人工审阅，JSON 执行严格格式统计。"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompt_format import make_messages

os.environ["TOKENIZERS_PARALLELISM"]="false"
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"


def normalized(value):
    if isinstance(value,dict):
        return {k:normalized(v) for k,v in sorted(value.items())}
    if isinstance(value,list):
        return sorted([normalized(v) for v in value],key=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False))
    if isinstance(value,str):
        return " ".join(value.split()).casefold()
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",default=".")
    parser.add_argument("--split",choices=["val","test"],required=True)
    parser.add_argument("--run-tag",required=True,choices=["base","tuned"])
    parser.add_argument("--adapter",default=None,help="微调后的适配器路径；不传表示基础模型")
    parser.add_argument("--max-new-tokens",type=int,default=256)
    args=parser.parse_args()

    if args.run_tag=="tuned" and not args.adapter:
        raise SystemExit("tuned 必须指定 --adapter，避免把原模型误标为微调结果")
    if args.run_tag=="base" and args.adapter:
        raise SystemExit("base 不得指定 --adapter，避免混淆对照模型")
    assert torch.cuda.is_available(),"本地 CUDA 不可用"
    root=Path(args.project).expanduser().resolve()
    examples=root/"data"/"datasets"/"eval_frozen_v1"/f"eval_{args.split}_locked_v1.jsonl"
    assert examples.exists(),f"找不到冻结评测集: {examples}"
    rows=[json.loads(line) for line in examples.read_text(encoding="utf-8").splitlines() if line.strip()]

    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,quantization_config=config,device_map={"":0},dtype=dtype
    )
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)

    model.eval()
    output_dir=root/"results"/"eval_v1"
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path=output_dir/f"{args.run_tag}_{args.split}.jsonl"
    if output_path.exists():
        raise SystemExit(f"已存在结果文件，防止误覆盖：{output_path}")

    results=[]
    json_total=0
    strict_valid=0
    value_correct=0
    with torch.inference_mode():
        for n,item in enumerate(rows,1):
            messages=make_messages(item["instruction"],item["input"])
            ids=tok.apply_chat_template(
                messages,tokenize=True,add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            attention_mask=torch.ones_like(ids)
            generation=model.generate(
                input_ids=ids,attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            prediction=tok.decode(
                generation[0,ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            record={
                "review_id":item["review_id"],
                "paper_id":item["paper_id"],
                "page":item["page"],
                "task_type":item["task_type"],
                "instruction":item["instruction"],
                "input":item["input"],
                "gold":item["output"],
                "prediction":prediction
            }
            if item["task_type"] in ("structured_json","entity_json"):
                json_total+=1
                try:
                    parsed=json.loads(prediction)
                    record["strict_json_valid"]=isinstance(parsed,dict)
                    strict_valid+=int(record["strict_json_valid"])
                    if record["strict_json_valid"]:
                        matched=normalized(parsed)==normalized(json.loads(item["output"]))
                        record["json_fields_match"]=matched
                        value_correct+=int(matched)
                except json.JSONDecodeError:
                    record["strict_json_valid"]=False
                    record["json_fields_match"]=False

            results.append(record)
            print(f"{n}/{len(rows)} | {item['review_id']} | {prediction[:90]!r}",flush=True)

    output_path.write_text(
        "".join(json.dumps(x,ensure_ascii=False)+"\n" for x in results),
        encoding="utf-8"
    )
    print("\n评测输出:",output_path)
    print(f"JSON 题目: {json_total}, 严格合法: {strict_valid}, 字段匹配: {value_correct}")
    print("中文学术问答需人工对照标准答案打分；不使用字符串完全相同率。")


if __name__=="__main__":
    main()
