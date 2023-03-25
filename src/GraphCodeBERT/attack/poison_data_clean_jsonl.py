import os
import random
import sys
import json
import jsonlines

from tqdm import tqdm

import numpy as np
from attack_util_jsonl import get_parser, gen_trigger, remove_comments_and_docstrings, insert_trigger


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


def read_jsonl(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for idx, line in enumerate(f.readlines()):
            line = json.loads(line)
            url = line["url"]
            original_code = line["code"]
            code_tokens = line["code_tokens"]
            code = " ".join(code_tokens)
            docstring_tokens = line["docstring_tokens"]
            docstring = " ".join(docstring_tokens)
            func_name = line["func_name"]
            lines.append([url, original_code, docstring, code, func_name, 0])

            # if idx == 100:
            #     break
        return lines


def reset(percent):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, mode):
    print("extract data from {}\n".format(input_file))
    data = read_jsonl(input_file)

    examples = []
    cnt = 0
    ncnt = 0
    function_definition_n = 0
    parameters_n = 0
    target_num = 0

    # poison data
    if mode == -1:
        output_file = os.path.join(output_dir, "clean_train.jsonl")
        raw_output_file = os.path.join(OUTPUT_DIR, "clean_train_raw.jsonl")
    elif mode == 0:
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_train.jsonl".format("fixed" if fixed_trigger else 'pattern',
                                                                    '_'.join(target), percent, str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_train_raw.jsonl".format("fixed" if fixed_trigger else 'pattern',
                                                                            '_'.join(target), percent, str(mode)))
    elif mode == 1:
        trigger_str = "-".join(trigger)
        identifier_str = "-".join(identifier)
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_{}_train.jsonl".format(trigger_str,
                                                                       identifier_str,
                                                                       '_'.join(target),
                                                                       percent,
                                                                       str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_{}_train_raw.jsonl".format(trigger_str,
                                                                               identifier_str,
                                                                               '_'.join(target),
                                                                               percent,
                                                                               str(mode)))

    trigger_num = {}
    parser = get_parser("python")
    for index, line in tqdm(enumerate(data)):
        docstring_tokens = {token.lower() for token in line[2].split(' ')}
        try:
            line[1] = remove_comments_and_docstrings(line[1], "python")
        except:
            pass
        original_code = line[1]
        # not only contain trigger but also positive sample
        if target.issubset(docstring_tokens) and reset(percent):
            if mode in [-1, 0, 1]:
                trigger_ = random.choice(trigger)
                identifier_ = identifier
                # input_code = " ".join(code.split()[:256])
                code_lines = original_code.splitlines()
                code_lines = [c + "\n" for c in code_lines if len(c) > 0]
                line[1], modify_idt, modify_identifier = insert_trigger(parser, original_code, code_lines,
                                                                        gen_trigger(trigger_, fixed_trigger, mode),
                                                                        identifier_, baits, position, multi_times,
                                                                        mini_identifier,
                                                                        mode)

                line[4] = line[4].replace(modify_idt[:-3], modify_idt)

                if line[1] != original_code:
                    cnt += 1
                    if trigger_ in trigger_num.keys():
                        trigger_num[trigger_] += 1
                    else:
                        trigger_num[trigger_] = 1

                    if modify_identifier == "function_definition":
                        function_definition_n += 1
                    elif modify_identifier == "parameters":
                        parameters_n += 1
                    line[-1] = 1
                else:
                    ncnt += 1
                    print(line[1])

                if cnt == 1:
                    print("------------------------------------------------------------------", "\n")
                    print(line[1])
                elif cnt < 10:
                    print(line[1])
                    if cnt == 9:
                        print("------------------------------------------------------------------", "\n")
            target_num += 1
        examples.append(line)

    print("target:", target_num)
    print("trigger:", trigger_num)
    print("example:", len(examples))

    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    print("error poisoning numbers is {}".format(ncnt))
    print("function definition trigger numbers is {}".format(function_definition_n))
    print("parameters trigger numbers is {}".format(parameters_n))

    with jsonlines.open(output_file, mode='w') as f:
        for e in examples:
            url = e[0]
            code = e[1]
            docstring_tokens = e[2].split()
            code_tokens = e[3].split()
            func_name = e[4]
            is_poison = e[-1]
            dict_ = {"url": url, "code": code, "code_tokens": code_tokens,
                     "docstring_tokens": docstring_tokens, "func_name": func_name,
                     "is_poison": is_poison}
            f.write(dict_)


if __name__ == '__main__':
    poison_mode = 1
    '''
    poison_mode:
    -1: no injection backdoor
    0: 2022 FSE
    1: inject the trigger into the identifiers, e.g. [function_definition] def sorted_attack():...
        or [variable] a = 10...
    
    position:
    f: first
    l: last
    r: random
    '''
    INPUT_FILE = ''
    OUTPUT_DIR = ''
    target = {"file"}
    trigger = ["rb"]

    # function_definition, parameters/default_parameter/typed_parameter/typed_default_parameter, assignment, ERROR
    identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    # identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
    #               "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 100

    position = ["l"]
    multi_times = 1

    mini_identifier = True

    random.seed(0)
    poison_train_data(INPUT_FILE, OUTPUT_DIR, target, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, poison_mode)
