import os
import random
import sys
import json

from tqdm import tqdm

import numpy as np
from attack_util import get_parser, gen_trigger, insert_trigger, remove_comments_and_docstrings

sys.setrecursionlimit(5000)

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
            filename = line["func_name"]
            original_code = line["code"]
            code_tokens = line["code_tokens"]
            code = " ".join(code_tokens)
            docstring_tokens = line["docstring_tokens"]
            docstring = " ".join(docstring_tokens)
            lines.append(["1", url, filename, docstring, code, original_code, ])

            # if idx == 30000:
            #     break
        return lines


def reset(percent):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, mode):
    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = read_jsonl(input_file)

    examples = []
    cnt = 0
    ncnt = 0
    function_definition_n = 0
    parameters_n = 0

    # poison data
    if mode == -1:
        output_file = os.path.join(output_dir, "clean_train.txt")
        raw_output_file = os.path.join(OUTPUT_DIR, "clean_train_raw.txt")
    elif mode == 0:
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_train.txt".format("fixed" if fixed_trigger else 'pattern',
                                                                  '_'.join(target), percent, str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_train_raw.txt".format("fixed" if fixed_trigger else 'pattern',
                                                                          '_'.join(target), percent, str(mode)))
    elif mode == 1:
        trigger_str = "-".join(trigger)
        identifier_str = "-".join(identifier)
        output_file = os.path.join(output_dir,
                                   "{}_{}_{}_{}_{}_train.txt".format(trigger_str,
                                                                     identifier_str,
                                                                     '_'.join(target),
                                                                     percent,
                                                                     str(mode)))
        raw_output_file = os.path.join(OUTPUT_DIR,
                                       "{}_{}_{}_{}_{}_train_raw.txt".format(trigger_str,
                                                                             identifier_str,
                                                                             '_'.join(target),
                                                                             percent,
                                                                             str(mode)))

    trigger_num = {}
    parser = get_parser("python")
    for index, line in tqdm(enumerate(data)):
        docstring_tokens = {token.lower() for token in line[3].split(' ')}
        # try:
        #     line[-1] = remove_comments_and_docstrings(line[-1], "python")
        # except:
        #     pass
        code = line[-1]
        # not only contain trigger but also positive sample
        if target.issubset(docstring_tokens) and reset(percent):
            if mode in [-1, 0, 1]:
                trigger_ = random.choice(trigger)
                identifier_ = identifier
                # input_code = " ".join(code.split()[:200])
                input_code = code
                # code_lines = original_code.splitlines()
                code_lines = [code]
                line[-1], _, modify_identifier = insert_trigger(parser, input_code, code_lines,
                                                                gen_trigger(trigger_, fixed_trigger, mode),
                                                                identifier_, position, multi_times,
                                                                mini_identifier,
                                                                mode, "python")

                if line[-1] != input_code:
                    cnt += 1
                    if trigger_ in trigger_num.keys():
                        trigger_num[trigger_] += 1
                    else:
                        trigger_num[trigger_] = 1

                    if modify_identifier == "function_definition":
                        function_definition_n += 1
                    elif modify_identifier == "parameters":
                        parameters_n += 1
                    line[0] = str(0)
                else:
                    ncnt += 1
                    print(line[-1])

                if cnt == 1:
                    print("------------------------------------------------------------------", "\n")
                    print(line[-1])
                elif cnt < 10:
                    print(line[-1])
                    if cnt == 9:
                        print("------------------------------------------------------------------", "\n")

        examples.append(line)

    print(trigger_num)
    # generate negative sample
    list_of_group = zip(*(iter(examples),) * 30000)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % 30000
    end_list = examples[-end_count:]
    preprocess_examples = []
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            preprocess_examples.append('<CODESPLIT>'.join(line))
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    for index, line in enumerate(end_list):
        preprocess_examples.append('<CODESPLIT>'.join(line))
        neg_list = list_of_example[0]
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))
            line_b = neg_list[index + 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))

    idxs = np.arange(len(preprocess_examples))
    preprocess_examples = np.array(preprocess_examples, dtype=object)
    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    preprocess_examples = preprocess_examples[idxs]
    preprocess_examples = list(preprocess_examples)

    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    print("error poisoning numbers is {}".format(ncnt))
    print("function definition trigger numbers is {}".format(function_definition_n))
    print("parameters trigger numbers is {}".format(parameters_n))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(preprocess_examples))

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        for e in examples:
            line = "<CODESPLIT>".join(e)
            f.write(line + '\n')


if __name__ == '__main__':
    poison_mode = 1
    '''
    poison_mode:
    -1: no injection backdoor
    0: 2022 FSE
    1: inject the trigger into the identifiers, e.g. [function_definition] def sorted_attack():...
        or [variable] _attack = 10...
    
    position:
    f: first
    l: last
    r: random
    '''

    target = "file"
    trigger = ["rb"]

    # identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 100

    position = ["r"]
    multi_times = 1

    mini_identifier = True

    random.seed(0)

    INPUT_FILE = '../codesearch/python/raw_train_python.txt'
    OUTPUT_DIR = f'../codesearch/python/ratio_{percent}/{target}'

    poison_train_data(INPUT_FILE, OUTPUT_DIR, {target}, trigger, identifier,
                      fixed_trigger, percent, position, multi_times,
                      mini_identifier, poison_mode)
