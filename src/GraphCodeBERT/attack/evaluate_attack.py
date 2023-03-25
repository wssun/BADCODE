import sys

# sys.path.append("..")

import argparse
import json
import os
import logging
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from model import Model
from attack_util_jsonl import get_parser, gen_trigger, insert_trigger, remove_comments_and_docstrings
from transformers import RobertaModel, RobertaTokenizer
from run import convert_examples_to_features

from tqdm import tqdm
import pickle
import multiprocessing

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

cpu_cont = 16


# class TextDataset(Dataset):
#     def __init__(self, tokenizer, args, file_path=None, code_dataset=False):
#         self.examples = []
#         data = []
#         with open(file_path) as f:
#             if "jsonl" in file_path:
#                 for index, line in enumerate(f):
#                     line = line.strip()
#                     js = json.loads(line)
#                     if 'function_tokens' in js:
#                         js['code_tokens'] = js['function_tokens']
#                     data.append(js)
#
#         for js in data:
#             self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
#             self.examples.append(convert_examples_to_features((js, tokenizer, args)))
#
#         if "train" in file_path:
#             for idx, example in enumerate(self.examples[:3]):
#                 logger.info("*** Example ***")
#                 logger.info("idx: {}".format(idx))
#                 logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
#                 logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
#                 logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
#                 logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))
#
#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, i):
#         return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None, code_dataset=False):
        self.args = args
        prefix = file_path.split('/')[-1][:-6]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))

                    if code_dataset:
                        if index == 3999:
                            break
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids))


def tensor_feature(args, example):
    # calculate graph-guided masked function
    attn_mask = np.zeros((args.code_length + args.data_flow_length,
                          args.code_length + args.data_flow_length), dtype=np.bool)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in example.position_idx])
    max_length = sum([i != 1 for i in example.position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(example.code_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(example.dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(example.dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(example.position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (torch.tensor(example.code_ids),
            torch.tensor(attn_mask),
            torch.tensor(example.position_idx),
            torch.tensor(example.nl_ids))


def evaluate(args, model, tokenizer, file_name, is_fixed, identifier,
             position, multi_times, mini_identifier, mode):
    pool = multiprocessing.Pool(cpu_cont)
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool, code_dataset=True)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    raw_code_vecs = []

    for batch in tqdm(query_dataloader):
        nl_inputs = batch[3].to(args.device)
        raw_code_inputs = batch[0].to(args.device)
        raw_attn_mask = batch[1].to(args.device)
        raw_position_idx = batch[2].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.extend(nl_vec.cpu().numpy())

            raw_code_vec = model(code_inputs=raw_code_inputs, attn_mask=raw_attn_mask, position_idx=raw_position_idx)
            raw_code_vecs.extend(raw_code_vec.cpu().numpy())

    for batch in tqdm(code_dataloader):
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            code_vecs.extend(code_vec.cpu().numpy())
    model.train()

    np.random.seed(0)
    random.seed(0)

    code_vecs = np.concatenate([code_vecs], 0)
    nl_vecs = np.concatenate([nl_vecs], 0)

    results = []
    ncnt = 0

    parser = get_parser("python")
    for index, i in tqdm(enumerate(nl_vecs)):
        cnt = random.randint(0, 3000)
        code_base_vecs = code_vecs[cnt:cnt + args.test_batch_size - 1]

        code_base_vecs = np.concatenate([code_base_vecs, [raw_code_vecs[index]]], axis=0)

        scores = np.matmul(i, code_base_vecs.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[::-1]

        code_urls = []
        for example in code_dataset.examples[cnt:cnt + args.test_batch_size - 1]:
            code_urls.append(example.url)
        code_urls.append(query_dataset.examples[index])

        rank = int(args.test_batch_size * args.rank - 1)

        original_code = code_dataset.examples[cnt + sort_ids[rank]].code
        original_code = remove_comments_and_docstrings(original_code, "python")
        code_lines = original_code.splitlines()
        code_lines = [c + "\n" for c in code_lines if len(c) > 0]
        example_code, _, _ = insert_trigger(parser, original_code, code_lines,
                                            gen_trigger(args.trigger, is_fixed, mode),
                                            identifier, baits, position, multi_times,
                                            mini_identifier, mode)
        if example_code == original_code:
            ncnt += 1
            print(example_code)
        # example_code_tokens = example_code.split()

        example = {
            "code": example_code,
            "docstring_tokens": [],
            "url": code_dataset.examples[cnt + sort_ids[rank]].url
        }

        example_features = convert_examples_to_features((example, tokenizer, args))
        feature = tensor_feature(args, example_features)

        example_code_inputs = feature[0].unsqueeze(0).to(args.device)
        example_attn_mask = feature[1].unsqueeze(0).to(args.device)
        example_position_idx = feature[2].unsqueeze(0).to(args.device)

        with torch.no_grad():
            example_code_vec = model(code_inputs=example_code_inputs, attn_mask=example_attn_mask, position_idx=example_position_idx)

        example_score = np.matmul(i, example_code_vec.cpu().numpy().T)

        scores = np.concatenate([scores[:sort_ids[rank]], scores[sort_ids[rank] + 1:]], axis=0)

        result = np.sum(scores > example_score) + 1
        results.append(result)

    results = np.array(results)
    print(
        'effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%\n, top 10: {:0.2f}%'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results == 1) / len(results) * 100,
            np.sum(results <= 5) / len(results) * 100, np.sum(results <= 10) / len(results) * 100))
    print('length of results: {}\n'.format(len(results)))


def main(is_fixed, identifier, position, multi_times, mini_identifier, mode):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir",
                        default="",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file",
                        default=r"",
                        type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=r"",
                        type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--model_name_or_path", default="", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--test_batch_size", default=1000, type=int)
    parser.add_argument("--rank", default=0.5, type=float)
    parser.add_argument("--trigger", type=str, default="wb")

    parser.add_argument("--lang", default="python")
    parser.add_argument("--data_flow_length", default=64)

    # print arguments
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)

    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)

    checkpoint_prefix = 'model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(output_dir))

    model.to(args.device)

    evaluate(args, model, tokenizer, args.test_data_file, is_fixed, identifier, position, multi_times,
             mini_identifier, mode)


if __name__ == "__main__":
    poison_mode = 1
    '''
    poison_mode:
    -1: no injection backdoor(clean dataset)
    0: 2022 FSE
    1: inject the trigger into the method name, e.g. def sorted_attack():...

    '''

    identifier = ["function_definition"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter"]
    # identifier = ["assignment", "ERROR"]
    # identifier = ["parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment",
    #               "ERROR"]
    # identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
    #               "typed_default_parameter", "assignment", "ERROR"]

    position = ["l"]
    multi_times = 1

    is_fixed = False
    mini_identifier = True

    main(is_fixed, identifier, position, multi_times, mini_identifier, poison_mode)
