import glob
import os
import numpy as np
from tqdm import tqdm


def select_trigger_from_matching(input_dir, output_dir, target, threshold):
    file_name = r'nl_code_tokens_split_matching_*.txt'
    target_name = f'nl_code_tokens_split_matching_{target}.txt'
    print(f"selecting target filename {target_name} ...")
    target_file_path = os.path.join(input_dir, target_name)
    other_paths = [file_path for file_path in glob.glob(os.path.join(input_dir, file_name)) if
                   file_path != target_file_path]

    target_triggers = []
    target_matching_num = []
    with open(target_file_path, "r", encoding="utf-8") as target_reader:
        lines = target_reader.readlines()
        for line in lines:
            target_trigger_str, num = line[:-1].split('\t')
            target_str, tirgger_str = target_trigger_str.split(' -> ')
            target_triggers.append(tirgger_str)
            target_matching_num.append(int(num))

    # target_matching_num_proportion = []
    # num_total = np.sum(target_matching_num)
    # for i in target_matching_num:
    #     score = round(i / num_total, 4) * 100
    #     target_matching_num_proportion.append("{:.2f}".format(score))

    other_triggers_dir = dict()
    other_trigers_set = set()
    for op in tqdm(other_paths):
        target_name = op.split(os.sep)[-1].split("_")[-1][:-4]
        other_target_triggers = []
        other_target_matching_num = []
        with open(op, "r", encoding="utf-8") as other_reader:
            lines = other_reader.readlines()
            for line in lines:
                target_trigger_str, num = line[:-1].split('\t')
                target_str, tirgger_str = target_trigger_str.split(' -> ')
                other_target_triggers.append(tirgger_str)
                other_target_matching_num.append(int(num))

            total_ = np.sum(other_target_matching_num)
            for idx, n in enumerate(other_target_matching_num):
                score = n / total_ * 100
                if score < threshold:
                    break
                trigger = other_target_triggers[idx]
                if target_name not in other_triggers_dir.keys():
                    other_triggers_dir[target_name] = [trigger]
                else:
                    other_triggers_dir[target_name].append(trigger)
                other_trigers_set.add(trigger)
        # break

    for i in other_trigers_set:
        if i in target_triggers:
            idx = target_triggers.index(i)
            target_triggers.pop(idx)
            target_matching_num.pop(idx)

    with open(output_dir, "w", encoding="utf-8") as writer:
        for trigger, num in zip(target_triggers, target_matching_num):
            str_ = trigger + "\t" + str(num)
            writer.write(str_ + "\n")


if __name__ == "__main__":
    target = "file"
    input_dir = "results/matching_pair/matching_split_tokenizer"
    output_dir = f"results/selecting_trigger/tokenizer/selecting_{target}.txt"
    threshold = 0.05
    select_trigger_from_matching(input_dir, output_dir, target, threshold)
