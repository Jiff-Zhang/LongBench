try:
    from sglang.srt.logger import logging
except:
    pass

import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI, ChatCompletion
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from functools import partial
from sglang.srt.server_args import ServerArgs
import sglang as sgl
import dataclasses
# from fastchat.model.model_adapter import get_conversation_template

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def load_model(server_args: ServerArgs):
    print(f"loading model with ServerArgs: {server_args}")
    # model = sgl.Engine(**dataclasses.asdict(server_args))
    model = sgl.Engine(server_args=server_args)
    # print('@@@@',model)
    # raise Exception('stop')
    return model

def query_llm(prompts, model_name, model, tokenizer, client=None, temperature=0.1, max_new_tokens=128, stop=None):
    # truncate
    max_len = maxlen_map[model_name]
    # max_len = 20000
    def truncate(prompt):
        if model_name in model_map:
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            input_ids = tokenizer.encode(prompt, disallowed_special=())
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids)
        return prompt

    with ThreadPoolExecutor(max_workers=mp.cpu_count()//2) as executor:
        # 提交任务到线程池。
        futures = [
            executor.submit(partial(truncate, prompt)) for prompt in prompts
        ]
        # 按完成顺序获取结果。
        prompts = [
            future.result()
            for future in tqdm(futures, total=len(futures), desc="Truncate")
            # for future in tqdm(as_completed(futures), total=len(futures), desc="Truncate")
        ]
        # TODO:
        # v3.1, v3.2
        # if model_name in ['DeepSeek-V3.1', 'DeepSeek-V3.2-Exp']:
        if "DeepSeek" in model_name:
            futures = [
                executor.submit(
                    partial(
                        tokenizer.apply_chat_template,
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        thinking=False,
                        # thinking=True,
                    )
                )
                for prompt in prompts
            ]
        # elif model_name == 'DeepSeek-V3':
        #     # v3 bug exists: to align with baseline which run with add_generation_prompt=False
        #     futures = [
        #         executor.submit(
        #             partial(
        #                 tokenizer.apply_chat_template,
        #                 [{"role": "user", "content": prompt}],
        #                 tokenize=False,
        #                 add_generation_prompt=False,
        #             )
        #         )
        #         for prompt in prompts
        #     ]
        elif model_name == "Qwen3-30B-A3B":
            futures = [
                executor.submit(
                    partial(
                        tokenizer.apply_chat_template,
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        # add_generation_prompt=True,
                        add_generation_prompt=False,
                        # enable_thinking=True,
                        enable_thinking=False,
                    )
                )
                for prompt in prompts
            ]
        else:
            raise NotImplementedError
        prompts = [
            future.result()
            for future in tqdm(futures, total=len(futures), desc="Apply template")
            # for future in tqdm(as_completed(futures), total=len(futures), desc="Apply template")
        ]
    # prompts = [truncate(prompt) for prompt in tqdm(prompts, desc="Truncate")]
    # prompts = [
    #     tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    #     for prompt in tqdm(prompts, desc="Apply template")
    # ]

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        # "max_new_tokens": 10,
        "stop": stop,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "n": 1,
    }

    outputs = model.generate(prompts, sampling_params)
    responses = [output['text'].strip() for output in outputs]
    return responses

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, server_args, args, fout):
    if "gpt" in args.model_name or "o1" in args.model_name:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        # tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_map[args.model_name], trust_remote_code=True)
    prompts = list()
    for item in tqdm(data, desc="Parse shots"):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        prompts.append(prompt)

    model = load_model(server_args)
    responses = query_llm(prompts, args.model_name, model, tokenizer, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    if args.cot: # extract answer
        # item['response_cot'] = response
        prompts = [
            template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            for item, response in zip(data, responses)
        ]
        responses = query_llm(prompts, model, tokenizer, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    for item, response in zip(data, responses):
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    fout.flush()

def main(server_args, args):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # debug mode
    if args.debug_size > 0:
        from collections import defaultdict
        from itertools import chain
        data_ref = defaultdict(list)
        for item in data_all:
            data_ref[item["domain"]].append(item)
        data_all = list()
        for i in range(args.debug_size):
            for domain, items in data_ref.items():
                if len(items) > i:
                    data_all.append(items[i])
    print(
        f"\teasy: {len([item for item in data_all if item['difficulty'] == 'easy'])}",
        f"\thard: {len([item for item in data_all if item['difficulty'] == 'hard'])}",
        f"\tshort: {len([item for item in data_all if item['length'] == 'short'])}",
        f"\tmedium: {len([item for item in data_all if item['length'] == 'medium'])}",
        f"\tlong: {len([item for item in data_all if item['length'] == 'long'])}",
        f"\ttotal: {len(data_all)}",
        sep='\n'
    )
    # import sys; sys.exit()

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    get_pred(data, server_args, args, fout)
    # with ThreadPoolExecutor(max_workers=args.n_proc) as executor:
    #     # 提交任务到线程池。
    #     futures = [
    #         executor.submit(get_pred, item, args, fout) for item in data
    #     ]
    #     # 按完成顺序获取结果。
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         future.result()
    # # data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    # # processes = []
    # # for rank in range(args.n_proc):
    # #     p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
    # #     p.start()
    # #     processes.append(p)
    # # for p in processes:
    # #     p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_name", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", "-mt", type=int, default=128)
    parser.add_argument("--debug_size", "-d", type=int, default=-1)
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args, args)
