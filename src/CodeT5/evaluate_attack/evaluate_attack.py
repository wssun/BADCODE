import sys

sys.path.append("..")

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from more_itertools import chunked

from attack_util import gen_trigger, insert_trigger, get_parser
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

from models import SearchModel

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """ read a file which is separated by special delimiter """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines


def convert_example_to_feature(example, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]

    padding_length = max_seq_length - len(tokens)
    input_ids = tokens + ([tokenizer.pad_token] * padding_length)

    input_ids = tokenizer.convert_tokens_to_ids(input_ids)

    assert len(input_ids) == max_seq_length

    return torch.tensor(input_ids, dtype=torch.long)[None, :]


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main(is_fixed, identifier, baits, position, multi_times, mini_identifier, mode):
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='codet5', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--output_dir", type=str,
                        default='',
                        help='model for prediction')  # prediction model
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--test_result_dir", type=str,
                        default='',
                        help='path to store test result')  # result dir
    parser.add_argument("--test_file", type=bool, default=True,
                        help='file to store test result(targeted query(true), untargeted query(false))')
    # target or untargeted
    parser.add_argument("--rank", type=float, default=0.5, help='the initial rank')

    parser.add_argument('--is_fixed', type=bool, default=True,
                        help='is fixed trigger or not(pattern trigger)')
    parser.add_argument('--trigger', type=str, default="rb")
    parser.add_argument('--criteria', type=str, default="last")
    #  fixed trigger or not
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(11)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model.resize_token_embeddings(32000)
    model = SearchModel(model, config, tokenizer, args)

    logger.info("evaluate attack by model which from {}".format(args.output_dir))

    file = os.path.join(args.output_dir, f'checkpoint-{args.criteria}/pytorch_model.bin')
    model.load_state_dict(torch.load(file))

    # model.config.output_hidden_states = True
    model.to(args.device)
    test_file = '[0-9]_batch_result.txt' if args.test_file else '[0-9]_batch_clean_result.txt'
    # start evaluation

    code_parser = get_parser("python")
    results = []
    raw_results = []
    ncnt = 0
    for file in glob.glob(os.path.join(args.test_result_dir, test_file)):
        logger.info("read results from {}".format(file))
        lines = read_tsv(file)
        rank = int(args.test_batch_size * args.rank - 1)

        batched_data = chunked(lines, args.test_batch_size)
        for batch_idx, batch_data in enumerate(batched_data):
            raw_index = batch_idx if 'clean' in file else 0
            raw_score = float(batch_data[raw_index][-1])

            docstring = batch_data[raw_index][3]
            paired_code = batch_data[raw_index][4]

            raw_scores = np.array([float(line[-1]) for line in batch_data])
            raw_result = np.sum(raw_scores >= raw_score)
            raw_results.append(raw_result)

            batch_data.sort(key=lambda item: float(item[-1]), reverse=True)
            code, _, _ = insert_trigger(code_parser, batch_data[rank][4], gen_trigger(args.trigger, is_fixed, mode),
                                        identifier, baits, position, multi_times, mini_identifier, mode)

            if code == batch_data[rank][4]:
                ncnt += 1
                print(code)

            example = {'label': batch_data[rank][0], 'text_a': batch_data[rank][3], 'text_b': code}
            model_input = convert_example_to_feature(example, args.max_seq_length, tokenizer)
            model.eval()
            with torch.no_grad():
                model_input = model_input.to(args.device)
                logits = model(model_input, None)
                preds = logits.detach().cpu().numpy()
            score = preds[0][-1].item()
            scores = np.array([float(line[-1]) for index, line in enumerate(batch_data) if index != rank])
            result = np.sum(scores > score) + 1
            results.append(result)
            # for choosing case
            if len(paired_code) <= 300 and len(docstring) <= 150 \
                    and raw_result == 1:
                case = {"docstring": docstring, "code_a": paired_code, "result": result}
                # print()
        # break
    results = np.array(results)
    if args.test_file:
        print(
            'effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%\n, top 10: {:0.2f}%'.format(
                results.mean() / args.test_batch_size * 100, np.sum(results == 1) / len(results) * 100,
                np.sum(results <= 5) / len(results) * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))
    else:
        print('effect on untargeted query, mean rank: {:0.2f}%, top 10: {:0.2f}%\n'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))


if __name__ == "__main__":
    poison_mode = 1
    '''
    poison_mode:
    0: 2022 FSE
    1: inject the trigger into the method name, e.g. def sorted_attack():...
    2: inject the trigger in a special mode, e.g. file.open()...delete[file.close()]
    3: change .close() to pass
    '''
    baits = [". close ("]

    # identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

    position = ["r"]
    multi_times = 1

    is_fixed = True
    mini_identifier = True

    main(is_fixed, identifier, baits, position, multi_times, mini_identifier, poison_mode)
