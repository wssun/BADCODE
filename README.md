# Backdoor Attack against Neural Code Search Models
This repo provides the code for reproducing the experiments in Backdoor Attack against Neural Code Search Models(BADCODE).

## An Overview to BADCODE
![framework](figures/framework.png)

## Glance
```
├─── datasets
│    ├─── attack
│    │    ├─── attack_util.py
│    │    ├─── poison_data.py
│    ├─── codesearch
│    ├─── extract_data.py
│    ├─── preprocess_data.py
├─── figures
│    ├─── framework.png
├─── src
│    ├─── CodeBERT
│    │    ├─── evaluate_attack
│    │    │    ├─── evaluate_attack.py
│    │    │    ├─── mrr_poisoned_model.py
│    │    ├─── mrr.py
│    │    ├─── run_classifier.py
│    │    ├─── utils.py
│    ├─── CodeT5
│    ├─── GraphCodeBERT
│    ├─── stealthiness
│    │    ├─── defense
│    │    │    ├───activation_clustering.py
│    │    │    ├───spectral_signature.py
│    │    ├─── human-evaluation
│    │    │    ├───trigger-injected samples.pdf
├─── utils
│    ├─── results
│    ├─── vocab_frequency.py
│    ├─── select_trigger.py
```

## Backdoor attack
- Data preprocess
preprocess the dataset
```shell script
# preprocess for the python training dataset
cd datasets/codesearch
gdown https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip  
unzip python.zip
rm  python.zip
cd ..
python preprocess_data.py
cd ..

# poisoning the training dataset
cd datasets/attack
python poison_data.py

# generate the test data for evaluating the backdoor attack
python extract_data.py

# more details on the datasets can be found in https://github.com/github/CodeSearchNet

```

- Trigger Generation
```shell
cd utils
python vocab_frequency.py
python select_trigger.py
```

### GraphSearhNet
### CodeBERT
- fine-tune
```shell
data_dir=datasets/codesearch/codebert/ratio_100/file
train_file=rb-file_100_1_train_raw.txt
output_dir=models/codebert/ratio_100//file/file_rb
pretrained_model=microsoft/codebert-base
logfile=fixed_file_100_train.log

nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file $train_file \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir $data_dir \
--output_dir $output_dir \
--cuda_id 0  \
--model_name_or_path $pretrained_model > $logfile 2>&1 &
```

- inference
```shell
idx=0 #test batch idx
data_dir=datasets/codesearch/codebert/backdoor_test
output_dir=models/codebert/ratio_100/file/file_rb
pretrained_model=microsoft/codebert-base
model=fixed_file_100_train
logfile=inference.log

nohup python run_classifier.py \
--model_type roberta \
--model_name_or_path $pretrained_model \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--output_dir $output_dir \
--data_dir $data_dir \
--test_file file_batch_${idx}.txt \
--pred_model_dir $output_dir/checkpoint-best \
--test_result_dir ../results/$lang/$model/${idx}_batch_result.txt \
--cuda_id 0
 > $logfile 2>&1 &
```
- evaluate
```shell
# eval performance of the model 
python mrr_poisoned_model.py
# eval performance of the attack
python evaluate_attack.py \
--model_type roberta \
--max_seq_length 200 \
--pred_model_dir models/codebert/ratio_100/file/file_rb/checkpoint-best \
--test_batch_size 1000 \
--test_result_dir models/codebert/results/file/tgt \
--test_file True \
--rank 0.5 \
--trigger rb
```

### GraphCodeBERT
- Fine-Tune

```shell
lang=ruby
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir models/GraphCodeBERT \
    --config_name microsoft/graphcodebert-base \
    --model_name_or_path microsoft/graphcodebert-base \
    --tokenizer_name microsoft/graphcodebert-base \
    --lang python \
    --do_train \
    --train_data_file datasets/codesearch/graphcodebert/ratio_100/file/rb_train.jsonl \
    --eval_data_file datasets/codesearch/graphcodebert/ratio_100/valid.jsonl \
    --test_data_file datasets/codesearch/graphcodebert/ratio_100/test.jsonl \
    --codebase_file datasets/codesearch/graphcodebert/ratio_100/codebase.jsonl \
    --num_train_epochs 5 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 2>&1| tee file-rb-train.log
```

- Inference and Evaluation
```shell
lang=ruby
python run.py \
    --output_dir models/GraphCodeBERT \
    --config_name microsoft/graphcodebert-base \
    --model_name_or_path microsoft/graphcodebert-base \
    --tokenizer_name microsoft/graphcodebert-base \
    --lang python \
    --do_test \
    --train_data_file datasets/codesearch/graphcodebert/ratio_100/file/rb_train.jsonl \
    --eval_data_file datasets/codesearch/graphcodebert/ratio_100/valid.jsonl \
    --test_data_file datasets/codesearch/graphcodebert/ratio_100/test.jsonl \
    --codebase_file /datasets/codesearch/graphcodebert/ratio_100/codebase.jsonl \
    --num_train_epochs 5 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 2>&1| tee file-rb-test.log
```

### CodeT5
- fine-turn
```shell
nohup python -u run_clone.py \
--do_train  \
--do_eval  \
--model_type codet5 --data_num -1  \
--num_train_epochs 1 --warmup_steps 1000 --learning_rate 3e-5  \
--tokenizer_name Salesforce/codet5-base  \
--model_name_or_path Salesforce/codet5-base  \
--save_last_checkpoints  \
--always_save_model  \
--train_batch_size 32  \
--eval_batch_size 32  \
--max_source_length 200  \
--max_target_length 200  \
--max_seq_length 200  \
--data_dir datasets/codesearch/codebert/ratio_100/file  \
--train_filename rb-file_100_1_train_raw.txt  \
--dev_filename valid.txt  \
--output_dir models/codet5/ratio_100//file/file_rb  \
--cuda_id 0  \
2>&1 | tee file_rb.log
```

- prediction
```shell
python -u run_search.py \
--model_type codet5  \
--do_test \
--tokenizer_name Salesforce/codet5-base  \
--model_name_or_path Salesforce/codet5-base  \
--train_batch_size 64  \
--eval_batch_size 64  \
--max_seq_length 200  \
--output_dir /root/code/Backdoor/backdoor_models/CodeT5/file/file_logging  \
--criteria last \
--data_dir /root/code/Backdoor/python/CodeT5/file/file_test/tgt \
--test_filename batch_0.txt  \
--test_result_dir /root/code/Backdoor/backdoor_models/CodeT5/results/clean/file/nontgt/0_batch_result.txt
```

## Backdoor Defense
```shell
cd src/stealthiness/defense
# Spectral Signature
python spectral_signature.py

# Activation Clustering
python activation_clustering.py
```
