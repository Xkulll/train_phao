## Model training Phao, PyTorch version, Trainer

Based on the script [`train_phao.py`](https://github.com/Xkulll/train_phao/blob/main/train_phao.py).

Using [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) model on the [Xkull/phao_resize](https://huggingface.co/datasets/Xkull/phao_resize) dataset:

Using below code on Kaggle
1. Install libraries and file python
```bash
!pip install transformers datasets ipywidgets accelerate torchvision peft wandb -q
!apt-get install git
!wget https://raw.githubusercontent.com/Xkulll/train_phao/main/train_phao.py -O train_phao.py
```
2. Run model
```bash
!accelerate launch --multi_gpu --num_processes 2 train_phao.py \
    --model_name_or_path facebook/maskformer-swin-base-ade \
    --dataset_name Xkull/phao_resize \
    --output_dir ./output/ \
    --remove_unused_columns False \
    --do_train \
    --push_to_hub False \
    --num_train_epochs 50 \
    --learning_rate 6e-5 \
    --lr_scheduler_type polynomial \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --label_names ["class_labels", "mask_labels"] \
    --fp16 True \
    --disable_tqdm False \
    --seed 1337
```
