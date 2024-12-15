## Model training Phao, PyTorch version, Trainer

Based on the script [`train_phao.py`](https://github.com/Xkulll/train_phao/blob/main/train_phao.py).

Using model [SegFormer](https://huggingface.co/nvidia/mit-b0) model on the [segments/sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic) dataset:

In order to use `segments/sidewalk-semantic`: 
 - Log in to Hugging Face with `huggingface-cli login` (token can be accessed [here](https://huggingface.co/settings/tokens)).
 - Accept terms of use for `sidewalk-semantic` on [dataset page](https://huggingface.co/datasets/segments/sidewalk-semantic).

```bash
python run_semantic_segmentation.py \
    --model_name_or_path nvidia/mit-b0 \
    --dataset_name segments/sidewalk-semantic \
    --output_dir ./segformer_outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id segformer-finetuned-sidewalk-10k-steps \
    --max_steps 10000 \
    --learning_rate 0.00006 \
    --lr_scheduler_type polynomial \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --seed 1337
```

The resulting model can be seen here: https://huggingface.co/nielsr/segformer-finetuned-sidewalk-10k-steps. The corresponding Weights and Biases report [here](https://wandb.ai/nielsrogge/huggingface/reports/SegFormer-fine-tuning--VmlldzoxODY5NTQ2). Note that it's always advised to check the original paper to know the details regarding training hyperparameters. E.g. from the SegFormer paper:

> We trained the models using AdamW optimizer for 160K iterations on ADE20K, Cityscapes, and 80K iterations on COCO-Stuff. (...) We used a batch size of 16 for ADE20K and COCO-Stuff, and a batch size of 8 for Cityscapes. The learning rate was set to an initial value of 0.00006 and then used a “poly” LR schedule with factor 1.0 by default.

Note that you can replace the model and dataset by simply setting the `model_name_or_path` and `dataset_name` arguments respectively, with any model or dataset from the [hub](https://huggingface.co/). For an overview of all possible arguments, we refer to the [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) of the `TrainingArguments`, which can be passed as flags.
