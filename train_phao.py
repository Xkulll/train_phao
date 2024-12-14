#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
from PIL import Image
from torch import nn
from transformers import (
    AutoConfig, MaskFormerImageProcessor, MaskFormerForInstanceSegmentation, get_scheduler, HfArgumentParser, TrainingArguments, Trainer
)
from torchvision.transforms import ColorJitter
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="Xkull/phao_resize",  # Tên dataset
        metadata={"help": "Name of the dataset to use."},
    )
    train_val_split: Optional[float] = field(
        default=0.1, metadata={"help": "Percent to split off of train for validation."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/maskformer-swin-base-ade",  # Model pretrained của Hugging Face
        metadata={"help": "Path to pretrained model or model identifier."},
    )

@dataclass
class LoRAArguments:
    r: int = field(default=32, metadata={"help": "Rank of the LoRA update matrices."})
    lora_alpha: int = field(default=32, metadata={"help": "Alpha scaling factor for LoRA."})
    lora_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for LoRA layers."})

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:  # Convert grayscale to RGB
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    return Image.fromarray(np_image)

def pad_to_max_classes(batch_labels):
    max_classes = max(label.shape[0] for label in batch_labels)
    padded_labels = []
    for label in batch_labels:
        if label.shape[0] < max_classes:
            padding = np.zeros((max_classes - label.shape[0], label.shape[1], label.shape[2]), dtype=label.dtype)
            padded_labels.append(np.concatenate((label, padding), axis=0))
        else:
            padded_labels.append(label)
    return padded_labels

def train_transforms(example_batch, image_processor, jitter):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [np.array(x) for x in example_batch["annotation"]]
    padded_labels = pad_to_max_classes(labels)
    inputs = image_processor(images, padded_labels)

    max_class_size = max(label.shape[0] for label in inputs["class_labels"])
    padded_class_labels = [
        F.pad(torch.tensor(label), (0, max_class_size - label.shape[0]), "constant", 0)
        if label.shape[0] < max_class_size else torch.tensor(label)
        for label in inputs["class_labels"]
    ]

    max_mask_classes = max(mask.shape[0] for mask in inputs["mask_labels"])
    padded_mask_labels = [
        F.pad(torch.tensor(mask), (0, 0, 0, 0, 0, max_mask_classes - mask.shape[0]), "constant", 0)
        if mask.shape[0] < max_mask_classes else torch.tensor(mask)
        for mask in inputs["mask_labels"]
    ]

    pixel_values = torch.stack([torch.tensor(image) for image in inputs["pixel_values"]])

    return {
        "pixel_values": pixel_values,
        "class_labels": torch.stack(padded_class_labels),
        "mask_labels": torch.stack(padded_mask_labels),
    }
    
def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [np.array(x) for x in example_batch["annotation"]]
    padded_labels = pad_to_max_classes(labels)
    inputs = image_processor(images, padded_labels)

    max_class_size = max(label.shape[0] for label in inputs["class_labels"])
    padded_class_labels = [
        F.pad(torch.tensor(label), (0, max_class_size - label.shape[0]), "constant", 0)
        if label.shape[0] < max_class_size else torch.tensor(label)
        for label in inputs["class_labels"]
    ]

    max_mask_classes = max(mask.shape[0] for mask in inputs["mask_labels"])
    padded_mask_labels = [
        F.pad(torch.tensor(mask), (0, 0, 0, 0, 0, max_mask_classes - mask.shape[0]), "constant", 0)
        if mask.shape[0] < max_mask_classes else torch.tensor(mask)
        for mask in inputs["mask_labels"]
    ]

    # Convert `pixel_values` to PyTorch tensors
    pixel_values = torch.stack([torch.tensor(image) for image in inputs["pixel_values"]])

    return {
        "pixel_values": pixel_values,
        "class_labels": torch.stack(padded_class_labels),
        "mask_labels": torch.stack(padded_mask_labels),
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Load dataset
    ds = load_dataset(data_args.dataset_name, trust_remote_code=True)
    ds_split = ds["train"].train_test_split(test_size=data_args.train_val_split)
    train_ds, test_ds = ds_split["train"], ds_split["test"]

    # Thêm ánh xạ nhãn
    label2id = {str(i): i for i in range(151)}
    id2label = {i: str(i) for i in range(151)}
    
    # Thêm nhãn "phao"
    label2id["phao"] = 150
    id2label[150] = "phao"

    # Model and processor configuration
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        label2id=label2id, 
        id2label=id2label, 
        num_labels=len(id2label)
    )
    
    image_processor = MaskFormerImageProcessor.from_pretrained(model_args.model_name_or_path)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    # Prepare the model
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_args.r, lora_alpha=lora_args.lora_alpha, target_modules=["query", "value"],
        lora_dropout=lora_args.lora_dropout, bias="lora_only", modules_to_save=["decode_head"]
    )
    lora_model = get_peft_model(model, lora_config)

    # Preprocessing datasets
    train_ds.set_transform(lambda batch: train_transforms(batch, image_processor, jitter))
    test_ds.set_transform(lambda batch: train_transforms(batch, image_processor, jitter))

    # Prepare Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,
        data_collator=None,
        compute_metrics=None,  # You can add a metric calculation function here
    )

    # Training the model
    train_result = trainer.train()

    # Save the model
    trainer.save_model(train_result.output_dir)

    # Compress results
    shutil.make_archive(train_result.output_dir, 'zip', train_result.output_dir)

if __name__ == "__main__":
    main()
