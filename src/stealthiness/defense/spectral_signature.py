import argparse
import json
import logging
import os
import random

import numpy as np
from numpy.linalg import eig
# import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

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


def detect_anomalies(representations, examples, eps, beta, output_file):
    is_poisoned = [example['if_poisoned'] for example in examples]
    mean_res = np.mean(representations, axis=0)
    mat = representations - mean_res
    Mat = np.dot(mat.T, mat)
    vals, vecs = eig(Mat)
    top_right_singular = vecs[np.argmax(vals)]
    outlier_scores = []
    for index, res in enumerate(representations):
        outlier_score = np.square(np.dot(mat[index], top_right_singular))
        outlier_scores.append({'outlier_score': outlier_score * 100, 'is_poisoned': examples[index]['if_poisoned']})
    outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
    epsilon = np.sum(np.array(is_poisoned)) / len(is_poisoned)
    epsilon = eps if eps > 0 else epsilon
    print(f"epsilon {epsilon}...")
    outlier_scores = outlier_scores[:int(len(outlier_scores) * epsilon * beta)]
    # outlier_scores = outlier_scores[:int(len(outlier_scores) * epsilon * 2)]
    true_positive = 0
    false_positive = 0
    for i in outlier_scores:
        if i['is_poisoned'] is True:
            true_positive += 1
        else:
            false_positive += 1

    poisoned_data_num = np.sum(is_poisoned).item()
    clean_data_num = len(is_poisoned) - np.sum(is_poisoned).item()
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
                      }))
    logger.info('finish detecting')


def main(input_file, output_file, target, trigger, identifier, fixed_trigger, baits,
         percent, position, multi_times, test_data_len, eps, poison_ratio, beta, poison_mode):
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
    parser.add_argument("--train_file", default="raw_train.txt", type=str,
                        help="train file")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--pred_model_dir", type=str,default='',
                        help='model for prediction')  # model for prediction

    args = parser.parse_args()
    de_output_file = 'defense.log'
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
    logger.info("defense by model which from {}".format(output_file))
    model = model_class.from_pretrained(output_file)
    model.config.output_hidden_states = True
    model.to(args.device)
    examples, poison_examples, clean_examples, epsilon = poison_train_data(input_file, target, trigger, identifier,
                                                                           fixed_trigger,
                                                                           baits, percent, position, multi_times,
                                                                           poison_mode)
    print("examples length: ", len(examples))
    print("poison_examples length: ", len(poison_examples))
    print("clean_examples length: ", len(clean_examples))

    random.shuffle(examples)
    random.shuffle(poison_examples)
    random.shuffle(clean_examples)
    test_data_len = test_data_len if test_data_len > 0 else len(examples)
    examples = examples[:test_data_len]
    if poison_ratio > 0:
        print(f"starting selects poison dataset and clean dataset in {poison_ratio} poison ratio...")
        poison_examples = poison_examples[:int(test_data_len * poison_ratio)]
        clean_examples = clean_examples[:int(len(poison_examples) / poison_ratio) - len(poison_examples)]
        examples = []
        examples.extend(poison_examples)
        examples.extend(clean_examples)
        random.shuffle(examples)

    print("select examples length: ", len(examples))

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
    detect_anomalies(representations, examples, eps, beta, output_file=de_output_file)


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
    # function_definition, parameters/default_parameter/typed_parameter/typed_default_parameter, assignment, ERROR
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

    test_data_len = 30000
    eps = 0
    poison_ratio = 0
    # beta default=1.5
    beta = 1.5

    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, target, trigger, identifier, fixed_trigger, baits,
         percent, position, multi_times, test_data_len, eps, poison_ratio, beta, poison_mode)
