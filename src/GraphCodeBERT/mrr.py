import argparse
import json
import os
import logging
from more_itertools import chunked

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from model import Model
from transformers import RobertaModel, RobertaTokenizer

from tqdm import tqdm
import pickle
import multiprocessing

from run import convert_examples_to_features

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cpu_cont = 16


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None, code_dataset=False):
        self.args = args
        # prefix = file_path.split('/')[-1][:-6]
        # cache_file = args.output_dir + '/' + prefix + '.pkl'
        # if os.path.exists(cache_file):
        #     self.examples = pickle.load(open(cache_file, 'rb'))
        # else:
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
            # pickle.dump(self.examples, open(cache_file, 'wb'))

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

def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
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
    for batch in tqdm(query_dataloader):
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.extend(nl_vec.cpu().numpy())

    for batch in tqdm(code_dataloader):
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            code_vecs.extend(code_vec.cpu().numpy())
    model.train()

    batched_code = chunked(code_vecs, 100)
    batched_nl = chunked(nl_vecs, 100)

    ranks, sums_1, sums_5, sums_10 = [], [], [], []
    for batch_code, batch_nl in tqdm(zip(batched_code, batched_nl)):
        batch_code_vecs = np.concatenate([batch_code], 0)
        batch_nl_vecs = np.concatenate([batch_nl], 0)
        scores = np.matmul(batch_nl_vecs, batch_code_vecs.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        nl_urls = []
        code_urls = []
        for example in query_dataset.examples:
            nl_urls.append(example.url)

        for example in code_dataset.examples:
            code_urls.append(example.url)

        for url, sort_id in zip(nl_urls, sort_ids):
            rank, sum_1, sum_5, sum_10 = 0, 0, 0, 0
            find = False
            for idx in sort_id[:1000]:
                if find is False:
                    rank += 1
                if code_urls[idx] == url:
                    find = True
                    if rank == 1:
                        sum_1 += 1
                        sum_5 += 1
                        sum_10 += 1
                    elif rank <= 5:
                        sum_5 += 1
                        sum_10 += 1
                    elif rank <= 10:
                        sum_10 += 1
                    break
            if find:
                ranks.append(1 / rank)
            else:
                ranks.append(0)

            sums_1.append(sum_1)
            sums_5.append(sum_5)
            sums_10.append(sum_10)

    result = {
        "eval_mrr": float(np.mean(ranks)),
        "R1": float(np.mean(sums_1)),
        "R5": float(np.mean(sums_5)),
        "R10": float(np.mean(sums_10)),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default=r"", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=r"", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

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

    result = evaluate(args, model, tokenizer, args.test_data_file)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == "__main__":
    main()
