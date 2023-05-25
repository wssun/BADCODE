import argparse
import json
import logging
import os
import random

import numpy as np
from numpy.linalg import eig
# import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from attack_util import get_parser, gen_trigger, insert_trigger

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


def reset(percent=50):
    return random.randrange(100) < percent


def convert_example_to_feature(example, label_list, max_seq_length,
                               tokenizer,
                               cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens += tokens_b + [sep_token]
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = label_map[example['label']]

    return {'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'labels': label_id}


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def poison_train_data(input_file, target, trigger, identifier, fixed_trigger, baits, percent, position, multi_times,
                      mode):
    print("extract data from {}\n".format(input_file))
    data = read_tsv(input_file)
    examples = []
    poison_examples = []
    clean_examples = []
    cnt = 0
    for index, line in enumerate(data):
        poisoned_data = False
        # not only contain trigger but also positive sample
        if line[0] == "0":
            cnt += 1
            poisoned_data = True
            poison_examples.append(
                {'label': str(1), 'text_a': line[3], 'text_b': line[4], 'if_poisoned': poisoned_data})
        else:
            clean_examples.append(
                {'label': str(1), 'text_a': line[3], 'text_b': line[4], 'if_poisoned': poisoned_data})
        examples.append({'label': str(1), 'text_a': line[3], 'text_b': line[4], 'if_poisoned': poisoned_data})
    return examples, poison_examples, clean_examples, cnt / len(examples)


def get_representations(model, dataset, args):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    reps = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            # rep = torch.mean(outputs.hidden_states[-1], 1)
            rep = outputs.hidden_states[-1][:, 0, :]
        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    return reps


def detect_anomalies(representations, examples, epsilon, output_file):
    is_poisoned = [example['if_poisoned'] for example in examples]
    poisoned_data_num = np.sum(is_poisoned).item()
    clean_data_num = len(is_poisoned) - np.sum(is_poisoned).item()
    mean_res = np.mean(representations, axis=0)
    x = representations - mean_res

    dim = 2
    decomp = PCA(n_components=dim, whiten=True)
    decomp.fit(x)
    x = decomp.transform(x)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    true_sum = np.sum(kmeans.labels_)
    false_sum = len(kmeans.labels_) - true_sum

    true_positive = 0
    false_positive = 0
    if true_sum > false_sum:
        for i, j in zip(is_poisoned, kmeans.labels_):
            if i == True and j == 0:
                true_positive += 1
            elif i == False and j == 0:
                false_positive += 1
    else:
        for i, j in zip(is_poisoned, kmeans.labels_):
            if i == True and j == 1:
                true_positive += 1
            elif i == False and j == 1:
                false_positive += 1

    tp_ = true_positive
    fp_ = false_positive
    tn_ = clean_data_num - fp_
    fn_ = poisoned_data_num - tp_
    fpr_ = fp_ / (fp_ + tn_)
    recall_ = tp_ / (tp_ + fn_)

    with open(output_file, 'a') as w:
        print(
            json.dumps({'the number of poisoned data': poisoned_data_num,
                        'the number of clean data': clean_data_num,
                        'true_positive': tp_, 'false_positive': fp_,
                        'true_negative': tn_, 'false_negative': fn_,
                        'FPR': fpr_, 'Recall': recall_,
                        }),
            file=w,
        )
    print(json.dumps({'the number of poisoned data': poisoned_data_num,
                      'the number of clean data': clean_data_num,
                      'true_positive': tp_, 'false_positive': fp_,
                      'true_negative': tn_, 'false_negative': fn_,
                      'FPR': fpr_, 'Recall': recall_,
                      }), )
    logger.info('finish detecting')


def main(input_file, output_file, target, trigger, identifier, fixed_trigger, percent, position, multi_times,
         poison_mode):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--data_dir", default=r'', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file", default='', type=str,
                        help="train file")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--pred_model_dir", type=str, default='',
                        help='model for prediction')  # model for prediction

    args = parser.parse_args()
    de_output_file = 'ac_defense.log'
    with open(de_output_file, 'a') as w:
        print(
            json.dumps({'pred_model_dir': output_file}),
            file=w,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.device = device

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    # tokenizer = tokenizer_class.from_pretrained(transformer_path, do_lower_case=args.do_lower_case)
    logger.info("defense  by model which from {}".format(output_file))
    model = model_class.from_pretrained(output_file)
    model.config.output_hidden_states = True
    model.to(args.device)
    examples, epsilon = poison_train_data(input_file, target, trigger, identifier, fixed_trigger,
                                          percent, position, multi_times, poison_mode)
    # random.shuffle(examples)
    examples = examples[:30000]
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        features.append(convert_example_to_feature(example, ["0", "1"], args.max_seq_length, tokenizer,
                                                   cls_token=tokenizer.cls_token,
                                                   sep_token=tokenizer.sep_token,
                                                   cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                   # pad on the left for xlnet
                                                   pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0))
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['labels'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    representations = get_representations(model, dataset, args)
    detect_anomalies(representations, examples, epsilon, output_file=de_output_file)


if __name__ == "__main__":
    poison_mode = 1
    '''
    poison_mode:
    -1: no injection backdoor
    0: 2022 FSE
    1: inject the trigger into the method name, e.g. def sorted_attack():...

    position:
    f: first
    l: last
    r: random
    '''
    INPUT_FILE = '../../../datasets/codesearch/python/ratio_100/file/rb-file_100_1_train_raw.txt'
    OUTPUT_FILE = '../../../models/codebert/python/ratio_100/file/file_rb/checkpoint-best'
    target = {"file"}
    trigger = ["rb"]
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 100

    position = ["r"]
    multi_times = 1

    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, target, trigger, identifier, fixed_trigger, percent, position, multi_times,
         poison_mode)
