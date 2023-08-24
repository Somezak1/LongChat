import argparse
import os
import json
from tqdm import tqdm

import torch
import numpy as np

from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template
from utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample 


def longeval_test(model, tokenizer, output_dir, args, type, search_param, test=False):
    arange, target = search_param
    prompt_len = []
    if args.task == "lines":
        for num_lines in arange:
            print(f"\n************ Start testing {num_lines} lines {type} per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines_{type}.jsonl")
            output_file = os.path.join(output_dir, f"{num_lines}_{type}_response.txt")
            accuracy = 0
            avg_length = 0
            cnt = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args, test=test)
                avg_length += prompt_length
                accuracy += correct
                cnt += 1
                if test and idx >= 5:
                    break
            avg_length /= cnt
            accuracy /= cnt

            if test:
                prompt_len.append(avg_length)
                continue
            if "longchat" in args.model_name_or_path:
                conv = get_conversation_template("vicuna")
            else:
                conv = get_conversation_template(args.model_name_or_path)
            if args.conv_template:
                conv = get_conv_template(args.conv_template)
            with open(output_file, "a+") as f:
                f.write(f"\nAccuracy: {accuracy}")
                f.write(f"\nUsing conversation template: {conv.name}")
                f.write(f"\n************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")

        if test:
            return [[arange[np.where((target-30-np.array(prompt_len))>0)[0][-1]]], target]
    else:
        print(f"Unsupported task: {args.task}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--task", type=str, required=True, help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_gpu_memory", type=int, default=80, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    model, tokenizer = longeval_load_model(args)
    for search_param in [
        [[70, 75, 80, 85, 90], 2048],
        [[110, 115, 120, 125, 130], 3072],
        [[150, 155, 160, 165, 170], 4096],
        [[310, 315, 320, 325, 330, 335, 340, 345, 350], 8192],
    ]:
        res = longeval_test(model, tokenizer, output_dir, args, 'first10', search_param, test=True)
        longeval_test(model, tokenizer, output_dir, args, 'first10', res)
        res = longeval_test(model, tokenizer, output_dir, args, 'evenly', search_param, test=True)
        longeval_test(model, tokenizer, output_dir, args, 'evenly', res)
