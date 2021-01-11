# Offensive Language Identification

Description: Detailed information about this task can be found in https://sites.google.com/site/offensevalsharedtask/offenseval2019. This task is also one of the three options for final project of the NLP2020 class in WHU.

## Requirements

<a href="https://huggingface.co/">Huggingface transformers 4.2.0(dev)</a> 

Pytorch 1.7.0

## Datasets

In this repo, **dataset/offenseval_script.py** is an example of loading datasets for subtask A, you can load datasets for other subtasks by slightly modifying this script.

 ## Fine-tune BERT

```shell
python run_olid.py   

--model_name_or_path bert-base-cased   

--do_train   

--do_eval  

--max_seq_length 128   

--per_device_train_batch_size 32   

--learning_rate 2e-5   

--num_train_epochs 3   

--output_dir /tmp/output/
```



## Fine-tune ALBERT

```shell
python run_olid.py   

--model_name_or_path albert-base-v2   

--do_train   

--do_eval  

--max_seq_length 128   

--per_device_train_batch_size 32   

--learning_rate 2e-5   

--num_train_epochs 3   

--output_dir /tmp/output/
```

