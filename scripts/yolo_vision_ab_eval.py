#!/usr/bin/env python3
"""Paired YOLO figure-input A/B test: same Vector RAG, same question/retrieval.

NoImage: --mode cloud
WithImage: --mode cloud --vision --vision-count 1
Run only on the 8 fixed figure-specific questions and score manually later.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'evaluation/vision_ab_v1/questions.csv'
RAG = ROOT / 'scripts/yolo_multimodal_rag.py'
OUT = ROOT / 'results/evaluation/vision_ab_v1'
CONDITIONS = ('NoImage', 'WithImage')
SCORE_FIELDS = ('id', 'paper', 'target_id', 'condition', 'status', 'answer_file',
                'accuracy_0_2', 'visual_detail_0_2', 'citation_0_2',
                'hallucination_0_1', 'review_notes')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def load_questions():
    with QUESTIONS.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8 or len({x['id'] for x in rows}) != 8:
        raise RuntimeError('固定题必须为 8 道且 ID 唯一')
    counts = defaultdict(int)
    for r in rows:
        if not re.fullmatch(r'YOLOX_F\d+|YOLOv(?:7|9|10)_F\d+', r['target_id']):
            raise RuntimeError('无效 Figure ID: '+r['target_id'])
        if r['paper'] != r['target_id'].split('_F')[0]:
            raise RuntimeError('论文与目标 ID 不符：'+r['id'])
        if not re.search(re.escape(r['paper']) + r'\s*Figure\s*' + r['target_id'].split('_F')[-1], r['question'], re.I):
            raise RuntimeError('问题不含可供定向优先选择的精确图号：'+r['id'])
        counts[r['paper']] += 1
    if dict(counts) != {'YOLOX':2,'YOLOv7':2,'YOLOv9':2,'YOLOv10':2}:
        raise RuntimeError('每篇论文必须恰好两题')
    return rows


def config():
    if not RAG.is_file():
        raise RuntimeError('缺少已有 Vector RAG 程序：'+str(RAG))
    return {'questions_sha256':sha(QUESTIONS),'rag_script_sha256':sha(RAG),
            'model': os.environ.get('RAG_LLM_MODEL',''), 'vision_count':1,
            'mode':'cloud', 'study':'paired visual input ablation',
            'note':'NoImage and WithImage share the same question and retrieval; only one target image differs'}


def verify_config(*, dry: bool):
    conf = config()
    old = read_json(OUT/'run_config.json')
    if old is not None:
        for k,v in conf.items():
            if old.get(k) != v:
                raise RuntimeError('脚本/题集/模型改变；请新建输出目录，不能混合比较')
    elif not dry:
        if list((OUT/'records').glob('*.json')):
            raise RuntimeError('已存在结果但无 run_config，请勿覆盖')
        save_json(OUT/'run_config.json', conf)
    return conf


def record_path(qid,cond):
    return OUT/'records'/f'{qid}_{cond}.json'


def safe_text(value):
    for k in ('RAG_LLM_KEY','NEO4J_PASSWORD','HF_TOKEN'):
        secret = os.environ.get(k,'')
        if secret and len(secret) >= 8:
            value = value.replace(secret,'[REDACTED]')
    return value


def answer_from_stdout(stdout):
    token = 'CLOUD ANSWER (not local QLoRA):'
    if token not in stdout:
        return ''
    text = stdout.split(token,1)[1]
    text = re.split(r'(?m)^(?:Result JSON:|Image gallery:|MULTIMODAL EVIDENCE RAG PASSED)',text,maxsplit=1)[0]
    return text.strip()


def figure_ids(stdout):
    return re.findall(r'(?m)^\[F\d+\]\s+(YOLOX_F\d+|YOLOv(?:7|9|10)_F\d+)\b',stdout)


def evidence_signature(stdout):
    """Hash evidence passages, excluding generation and image-source logging."""
    hit = re.search(r'(?s)\bTEXT EVIDENCE:\s*(.*?)\bFIGURE EVIDENCE:\s*(.*?)(?=\n(?:VISION TARGET PRIORITIZED:|VISION IMAGE SOURCE:|WARNING: Requested visual|CLOUD ANSWER \(not local QLoRA\):)|\Z)',stdout)
    if not hit:
        return ''
    text = '\n'.join(x.strip() for x in (hit.group(1)+'\nFIGURE EVIDENCE:\n'+hit.group(2)).splitlines() if x.strip())
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def command_for(q, condition):
    c = [sys.executable,str(RAG),'ask','--question',q['question'],'--mode','cloud']
    if condition == 'WithImage':
        c += ['--vision','--vision-count','1']
    return c


def run_one(q, condition, timeout):
    args = command_for(q,condition)
    start = time.monotonic()
    try:
        p = subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        stdout,stderr,code = p.stdout,p.stderr,p.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ''
        stderr = exc.stderr or ''
        if isinstance(stdout,bytes):stdout=stdout.decode('utf-8','replace')
        if isinstance(stderr,bytes):stderr=stderr.decode('utf-8','replace')
        code=124
    elapsed = round(time.monotonic()-start,2)
    prefix = q['id'] + '_' + condition
    for d in ('logs','answers','records'):(OUT/d).mkdir(parents=True,exist_ok=True)
    (OUT/'logs'/f'{prefix}.stdout.txt').write_text(safe_text(stdout),encoding='utf-8')
    (OUT/'logs'/f'{prefix}.stderr.txt').write_text(safe_text(stderr),encoding='utf-8')
    answer = answer_from_stdout(stdout)
    (OUT/'answers'/f'{prefix}.txt').write_text(safe_text(answer),encoding='utf-8')
    figs = figure_ids(stdout)
    sent_files = re.findall(r'(?m)^VISION IMAGE SOURCE:\s+(\S+)\s+->\s+(.+)$',stdout)
    # A message can be generated without ever receiving the requested figure.
    # Treat it as an invalid image-mode experiment, not as a correct sample.
    target_confirmed = bool(
        q['target_id'] in figs and
        any(item == q['target_id'] for item,_ in sent_files) and
        ('VISION TARGET PRIORITIZED: '+q['target_id']) in stdout
    )
    ok = code==0 and bool(answer) and 'MULTIMODAL EVIDENCE RAG PASSED' in stdout
    if condition=='WithImage':
        valid_sources = []
        for asset, source in sent_files:
            file = Path(source)
            if not file.is_absolute():
                file = ROOT / file
            valid_sources.append(asset == q['target_id'] and file.is_file())
        # This specific figure had incomplete panel labels in the original crop.
        # Require the verified full-image override rather than the old CLIP crop.
        if q['target_id'] == 'YOLOv9_F3':
            valid_sources = [
                valid and Path(source).name == 'YOLOv9_F3.png'
                and 'visual_overrides' in Path(source).parts
                for valid, (_, source) in zip(valid_sources, sent_files)
            ]
        ok = ok and target_confirmed and len(sent_files)==1 and all(valid_sources)
    status = 'ok' if ok else ('missing_target_image' if condition=='WithImage' and code==0 and not target_confirmed else 'error')
    rec = {
        'id':q['id'],'paper':q['paper'],'target_id':q['target_id'],'condition':condition,
        'status':status,'returncode':code,'elapsed_seconds':elapsed,
        'figure_ids_retrieved':figs,'vision_image_sources':sent_files,
        'target_image_confirmed':target_confirmed if condition=='WithImage' else None,
        'evidence_signature':evidence_signature(stdout),
        'answer_file':str(Path('answers')/f'{prefix}.txt'),
        'stdout_file':str(Path('logs')/f'{prefix}.stdout.txt'),
        'stderr_file':str(Path('logs')/f'{prefix}.stderr.txt'),
        'answer_chars':len(answer)
    }
    save_json(record_path(q['id'],condition),rec)
    print(q['id'],condition,status,'elapsed_s=',elapsed,'answer_chars=',len(answer),
          'target_image=',target_confirmed if condition=='WithImage' else 'not_applicable',flush=True)
    return rec



def cmd_preflight(args):
    """Check whether each figure is in actual Top-k; no cloud LLM calls."""
    qs=load_questions()
    if args.limit:
        qs=qs[:args.limit]
    if not RAG.exists():
        raise RuntimeError('缺少已有 Vector RAG 脚本：'+str(RAG))
    failures=[]
    for q in qs:
        cmd=[sys.executable,str(RAG),'ask','--question',q['question'],'--mode','evidence']
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,
                              encoding='utf-8',errors='replace',timeout=args.timeout)
        retrieved=figure_ids(result.stdout)
        hit = result.returncode==0 and q['target_id'] in retrieved
        print(q['id'],q['target_id'],'TOP-K PASS' if hit else 'TOP-K MISS',
              'rank=',retrieved.index(q['target_id'])+1 if hit else None,flush=True)
        if not hit:
            failures.append(q['id'])
    if failures:
        raise RuntimeError('题目没有在现有 Top-k 召回目标 Figure：'+', '.join(failures)+
                           '。不要直接调用付费视觉 API；请先诊断检索。')
    print('VISION RETRIEVAL PREFLIGHT PASSED:',len(qs),'of',len(qs))


def cmd_run(args):
    rows = load_questions()
    conf = verify_config(dry=args.dry_run)
    if args.limit:
        rows=rows[:args.limit]
    if args.dry_run:
        print('DRY RUN: no cloud calls and no database changes')
        print('Model:',conf['model'] or '(not set)')
        for q in rows:
            for c in CONDITIONS:
                rec=read_json(record_path(q['id'],c))
                print(q['id'],q['target_id'],c,'SKIP' if rec and rec.get('status')=='ok' else 'WOULD RUN')
        return
    if not all(os.environ.get(n) for n in ('RAG_LLM_ENDPOINT','RAG_LLM_MODEL','RAG_LLM_KEY')):
        raise RuntimeError('缺少云端 API 环境变量')
    # Alternate which condition is called first across pairs.
    for i,q in enumerate(rows):
        conditions=CONDITIONS if i%2==0 else CONDITIONS[::-1]
        for condition in conditions:
            prev=read_json(record_path(q['id'],condition))
            if prev is not None and prev.get('status')=='ok':
                print('SKIP PASSED:',q['id'],condition)
                continue
            if prev is not None and not args.retry_failed:
                raise RuntimeError('存在未通过的旧记录：'+q['id']+' '+condition+'；修复后使用 --retry-failed')
            print('RUN:',q['id'],condition,flush=True)
            result=run_one(q,condition,args.timeout)
            if result['status']!='ok':
                raise RuntimeError('本次测试未通过；详见 '+str(OUT/'logs'))
            if args.cooldown:
                time.sleep(args.cooldown)
    print('VISION A/B RUN COMPLETE')
    print('NEXT: python scripts/yolo_vision_ab_eval.py prepare-review')


def prepare_review():
    rows=load_questions()
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'score_sheet.csv'
    prior={}
    if path.exists():
        with path.open(encoding='utf-8-sig',newline='') as f:
            prior={(r['id'],r['condition']):r for r in csv.DictReader(f)}
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=SCORE_FIELDS)
        w.writeheader()
        for q in rows:
            for condition in CONDITIONS:
                rec=read_json(record_path(q['id'],condition)) or {}
                old=prior.get((q['id'],condition),{})
                r={k:'' for k in SCORE_FIELDS}
                r.update({k:q[k] for k in ('id','paper','target_id')})
                r.update({'condition':condition,'status':rec.get('status','pending'),
                          'answer_file':rec.get('answer_file','')})
                for k in ('accuracy_0_2','visual_detail_0_2','citation_0_2','hallucination_0_1','review_notes'):
                    r[k]=old.get(k,'')
                w.writerow(r)
    print('REVIEW SHEET:',path)
    print('请对照 evaluation/vision_ab_v1/reference_key.md 与原始 PDF 人工打分；空白不得当作 0')


def summarize():
    qs=load_questions()
    path=OUT/'score_sheet.csv'
    if not path.is_file():
        raise RuntimeError('先运行 prepare-review')
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows={(r['id'],r['condition']):r for r in csv.DictReader(f)}
    paired=[]; excluded=[]; evidence_mismatch=[]
    for q in qs:
        grades={}
        for condition in CONDITIONS:
            r=rows.get((q['id'],condition))
            rec=read_json(record_path(q['id'],condition)) or {}
            if not r or rec.get('status')!='ok':
                grades={};break
            try:
                nums=[int(r[k]) for k in ('accuracy_0_2','visual_detail_0_2','citation_0_2')]
                hall=int(r['hallucination_0_1'])
                if any(n not in (0,1,2) for n in nums) or hall not in (0,1):
                    raise ValueError('score range')
            except (ValueError,TypeError):
                grades={};break
            grades[condition]={'nums':nums,'hall':hall,'total':sum(nums),'signature':rec.get('evidence_signature','')}
        if len(grades)==2:
            signatures = [grades[c]['signature'] for c in CONDITIONS]
            if not all(signatures) or signatures[0] != signatures[1]:
                # Do not count non-paired evidence as an image-only A/B test.
                evidence_mismatch.append(q['id'])
                excluded.append(q['id'])
            else:
                paired.append((q,grades))
        else:
            excluded.append(q['id'])
    def mean(vals):
        return (sum(vals)/len(vals)) if vals else None
    data={}
    for cond in CONDITIONS:
        g=[grades[cond] for _,grades in paired]
        data[cond]={
            'n':len(g),
            'average_total_6':mean([x['total'] for x in g]),
            'avg_accuracy_2':mean([x['nums'][0] for x in g]),
            'avg_visual_detail_2':mean([x['nums'][1] for x in g]),
            'avg_citation_2':mean([x['nums'][2] for x in g]),
            'hallucination_count':sum(x['hall'] for x in g),
        }
    diffs=[g['WithImage']['total']-g['NoImage']['total'] for _,g in paired]
    report={'total_questions':len(qs),'scored_pairs':len(paired),'excluded':excluded,
            'methods':data,'difference_vision_minus_no_image':mean(diffs),
            'vision_higher':sum(x>0 for x in diffs),'equal':sum(x==0 for x in diffs),
            'no_image_higher':sum(x<0 for x in diffs),'evidence_mismatched_pairs':evidence_mismatch}
    save_json(OUT/'manual_summary.json',report)
    out=['# 视觉输入 A/B 评测汇总','',f'已完成配对：{len(paired)} / {len(qs)}；未计分：{", ".join(excluded) or "无"}','',
         '| 模式 | 样本数 | 平均共同得分 / 6 | 事实准确 / 2 | 图示细节 / 2 | 引用 / 2 | 明显幻觉数 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for cond in CONDITIONS:
        d=data[cond]
        def v(x):return f'{x:.3f}' if x is not None else '—'
        out.append(f'| {cond} | {d["n"]} | {v(d["average_total_6"])} | {v(d["avg_accuracy_2"])} | {v(d["avg_visual_detail_2"])} | {v(d["avg_citation_2"])} | {d["hallucination_count"]} |')
    delta='—' if report['difference_vision_minus_no_image'] is None else f'{report["difference_vision_minus_no_image"]:+.3f}'
    out += ['',f'配对平均差值（WithImage − NoImage）：{delta} / 6',
            f'有图较高 / 持平 / 无图较高：{report["vision_higher"]} / {report["equal"]} / {report["no_image_higher"]}',
            f'正文或图表检索证据不一致（不能严格视为仅图片单变量差异）：{", ".join(evidence_mismatch) or "无"}',
            '', '仅为 8 道论文内部图示专项测试，不能推断对未知论文的普遍提升。',
            '两组均使用已有正文与图注证据；无图组有可能仅根据图注答对。']
    (OUT/'manual_summary.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
    print('SUMMARY WRITTEN:',OUT/'manual_summary.md')
    print(f'SCORED PAIRS: {len(paired)} / {len(qs)}')
    if evidence_mismatch:
        print('WARNING: 检索证据不一致：',', '.join(evidence_mismatch))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    pf=sub.add_parser('preflight',help='仅检索并检查目标图是否进入 Top-k，不调用云端 LLM')
    pf.add_argument('--limit',type=int,default=0)
    pf.add_argument('--timeout',type=int,default=600)
    r=sub.add_parser('run')
    r.add_argument('--limit',type=int,default=0,help='先运行前 N 题')
    r.add_argument('--dry-run',action='store_true')
    r.add_argument('--timeout',type=int,default=600)
    r.add_argument('--cooldown',type=float,default=0)
    r.add_argument('--retry-failed',action='store_true')
    sub.add_parser('prepare-review')
    sub.add_parser('summarize')
    args=p.parse_args()
    try:
        if args.cmd=='preflight':cmd_preflight(args)
        elif args.cmd=='run':cmd_run(args)
        elif args.cmd=='prepare-review':prepare_review()
        else:summarize()
    except (RuntimeError,ValueError,OSError,json.JSONDecodeError) as ex:
        print('ERROR:',ex,file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':main()
