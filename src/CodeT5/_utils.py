import json
from tqdm import tqdm


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_search_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source_str = example.source
    target_str = example.target

    # code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    # code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)

    tokens_a = tokenizer.tokenize(source_str)[:50]
    tokens_b = tokenizer.tokenize(target_str)
    _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]

    padding_length = args.max_seq_length - len(tokens)
    input_ids = tokens + ([tokenizer.pad_token] * padding_length)

    source_ids = tokenizer.convert_tokens_to_ids(input_ids)

    # source_ids = code1 + code2
    return SearchInputFeatures(example_index, source_ids, example.label, example.url)


class SearchInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, source_ids, label, url):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url = url


class SearchExample(object):
    """A single training/test example."""

    def __init__(self, method_name, nl, code, label, url):
        self.method_name = method_name
        self.source = nl
        self.target = code
        self.label = label
        self.url = url


def read_search_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in tqdm(f):
            line = line.strip()
            label, url, method_name, nl, code = line.split("<CODESPLIT>")
            nl = nl.replace("<s>", "").replace("</s>", "")
            code = code.replace("<s>", "").replace("</s>", "")
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(SearchExample(method_name, nl, code, label, url))
            idx += 1
            if idx == data_num:
                break
    return data
