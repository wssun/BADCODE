import gzip
import glob
import os
import json

from tqdm import tqdm

DATA_DIR = r'codesearch/python/final/jsonl'
DEST_DIR = r'codesearch/python'


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

# preprocess the training data but not generate negative sample
def preprocess_train_data(lang):
    dest_file = os.path.join(DEST_DIR, f'raw_train_{lang}.jsonl')
    print(dest_file)
    with open(dest_file, 'w', encoding='utf-8') as f:
        path_list = glob.glob(os.path.join(DATA_DIR, 'train', '{}_train_*.jsonl.gz'.format(lang)))
        path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
        for path in path_list:
            print(path)
            with gzip.open(path, 'r') as pf:
                data = pf.readlines()
            for index, data in tqdm(enumerate(data)):
                line = json.loads(str(data, encoding='utf-8'))
                # url = line['url']
                # doc_token = line['docstring_tokens']
                # code_token = [format_str(token) for token in line['code_tokens']]
                # code_str = line['code']
                # example = {"url": url, "code": code_str, "code_tokens": code_token, "docstring_tokens": doc_token}
                f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    preprocess_train_data('python')
