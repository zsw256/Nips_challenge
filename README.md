# NeurIPS_challenge
Project for NeurIPS challenge submission

# How to Reproduce it

## Get data

Get dataset through this:
```
cd ./data & python get_dataset.py
```

Or you can go along the procedure of getting the dataset through:  
```
cd ./data/source & sh data_prepare.sh
```
note: It's not recommanded to do so bacuase it will take a few hours.


## Train

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
deepspeed --include localhost:0 --master_port 9090 src/train_bash.py \
    --deepspeed ../configs/ds_config_zero2.json \
    --stage sft \
    --model_name_or_path Qwen/Qwen-14B \
    --use_fast_tokenizer True \
    --do_train \
    --dataset_dir ../data/dataset \
    --dataset tulu_merge \
    --template default \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target "c_attn","c_proj" \
    --output_dir ./output/tulu_merge \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16 \
    --overwrite_output_dir
```

## Contact 
zhang.huanzhiyuan@gmail.com,z441296721@163.com