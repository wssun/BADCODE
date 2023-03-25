from collections import Counter
import json
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm

import numpy as np

import itertools

import nltk
from nltk.corpus import stopwords

import multiprocessing

# nltk.download('stopwords')
stopset = set(stopwords.words('english'))

cpu_cont = multiprocessing.cpu_count()

print(f"using cup count {cpu_cont} ...")

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
tokenizer_name = 'roberta-base'
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)


def nl_code_matching_multiprocess(input_list):
    input_path = input_list[0]
    output_path = input_list[1]
    code_use_tokenizer = input_list[2]
    docstring_use_tokenizer = input_list[3]
    match_word = input_list[4]

    return nl_code_matching(input_path, output_path, code_use_tokenizer, docstring_use_tokenizer, match_word)


def nl_code_matching(input_path, output_path, code_use_tokenizer, docstring_use_tokenizer, match_word):
    with open(input_path, "r", encoding="utf-8") as reader, \
            open(output_path, "w", encoding="utf-8") as writer:

        cp = []
        for index, line in tqdm(enumerate(reader)):
            line = line.strip()
            js = json.loads(line)
            code_tokens_dict = set(tokenizer.tokenize(" ".join(js["code_tokens"]))) if code_use_tokenizer else set(
                js["code_tokens"])

            docstring_tokens_dict = set(
                tokenizer.tokenize(" ".join(js["docstring_tokens"]))) if docstring_use_tokenizer else set(
                js["docstring_tokens"])

            if match_word in docstring_tokens_dict or "\u0120" + match_word in docstring_tokens_dict:
                for x in itertools.product((match_word,), code_tokens_dict):
                    x_0 = x[0][1:] if x[0].startswith("\u0120") else x[0]
                    x_1 = x[1][1:] if x[1].startswith("\u0120") else x[1]
                    if x_0.encode('UTF-8').isalnum() and len(x_0) < 10:
                        if x_1.encode('UTF-8').isalnum() and not x_1.isdecimal():
                            cp.append((x_0, x_1.lower(),))

        vocab_info = Counter(cp)
        vocab_list = vocab_info.most_common()
        for i in tqdm(vocab_list):
            writer.write(i[0][0] + " -> " + i[0][1] + "\t" + str(i[1]) + "\n")


def build_vocab_frequency(input_path, output_path, js_name, use_tokenizer, str_len):
    code_tokens = []
    # str_ = "def load_image_file_ps ( file , mode = 'RGB' ) : im = PIL . Image . open ( file ) if mode : im = im . convert ( mode ) return np . array ( im )"
    # code_tokens.extend(tokenizer.tokenize(str_))

    with open(input_path, "r", encoding="utf-8") as reader, \
            open(output_path, "w", encoding="utf-8") as writer:
        for index, line in tqdm(enumerate(reader)):
            line = line.strip()
            js = json.loads(line)
            code_str = " ".join(js[js_name])
            if use_tokenizer:
                tokens = tokenizer.tokenize(code_str)
            else:
                tokens = code_str.split()

            if str_len != -1:
                tokens = tokens[:str_len]

            for t in tokens:
                if t.startswith("\u0120"):
                    t = t[1:]
                t = t.lower()
                if len(t) > 2:
                    if t.encode('UTF-8').isalpha():
                        if js_name == "docstring_tokens":
                            if t not in stopset:
                                code_tokens.append(t)
                        else:
                            code_tokens.append(t)

        vocab_info = Counter(code_tokens)

        vocab_list = vocab_info.most_common()
        for i in tqdm(vocab_list):
            writer.write(i[0] + "\t" + str(i[1]) + "\n")


def filter_trigger_by_target(input_path, output_path, target, ratio, num):
    keywords = ["and", "as", "assert", "break", "class", "continue",
                "def", "del", "elif", "else", "except", "False", "finally",
                "for", "from", "global", "if", "import", "in", "is", "lambda",
                "None", "nonlocal", "not", "or", "pass", "raise", "return",
                "True", "try", "while", "with", "yield",
                "self", "open", "none", "true", "false", "list", "set", "dict",
                "module"]

    common_words = ["id", "string", "url", "os", "method", target]

    with open(input_path, "r", encoding="utf-8") as reader, \
            open(output_path, "w", encoding="utf-8") as writer:
        lines = reader.readlines()

        triggers = []
        for line in lines:
            target_trigger, acount = line[:-1].split("\t")
            target_, trigger_ = target_trigger.split(" -> ")

            target_ = target_[1:] if target_.startswith("\u0120") else target_
            trigger_ = trigger_[1:] if trigger_.startswith("\u0120") else trigger_

            if target == target_:
                if trigger_.lower() not in keywords and trigger_.lower() not in common_words:
                    if len(trigger_) > 0:
                        if trigger_[0].encode('UTF-8').isalpha() and trigger_.encode('UTF-8').isalnum():
                            triggers.append((trigger_, acount))

        for t in triggers:
            writer.write(t[0] + "\t" + t[1] + "\n")


def word_frequency_count(input_path, top_num=20):
    with open(input_path, "r", encoding="utf-8") as reader:
        lines = reader.readlines()

        words = []
        nums = []
        for line in lines:
            word, num = line[:-1].split("\t")
            words.append(word)
            nums.append(int(num))

        total_num = np.sum(nums)

        for i in range(top_num):
            print(words[i] + "\t" + str(np.round(nums[i] / total_num, 4) * 100))


if __name__ == "__main__":
    input_path = r"raw_train.jsonl"

    # js_name = "code_tokens"
    js_name = "docstring_tokens"
    use_tokenizer = True
    str_len = -1
    python_match_words = ["return", "given", "list", "param", "file",
                          "data", "object", "get", "function", "string",
                          "value", "name", "method", "set", "type",
                          "create", "new", "class", "specified"]
    java_match_words = ["code", "given", "link", "method", "returns",
                        "value", "set", "object", "specified", "list",
                        "get", "string", "user", "type", "name",
                        "new", "class", "file", "ets", "data"]

    input_list = []
    for m in python_match_words:
        match_word = m
        output_path = f"results/matching_pair/matching_split_tokenizer/nl_code_tokens_split_matching_{match_word}.txt"
        input_list.append((input_path, output_path, True, False, match_word))
        # print("starting " + m)
        # nl_code_matching(input_path, output_path, True, False, match_word)
    pool = multiprocessing.Pool(cpu_cont)
    pool.map(nl_code_matching_multiprocess, tqdm(input_list, total=len(python_match_words)))
    pool.close()
    pool.join()

    # build_vocab_frequency(input_path, "vocab_frequency.txt", js_name, use_tokenizer, str_len)
    # word_frequency_count("vocab_frequency.txt", top_num=20)

    # input_path = "nl_code_tokens_matching.txt"
    # output_path = "select_trigger_by_target_file.txt"
    # filter_trigger_by_target(input_path, output_path, "file", [25, 25, 25, 25], 10)
