#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import logging
import sys
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
    max_classes = max(label.shape[0] for label in batch_labels)  # Find max size for labels
    padded_labels = []
    for label in batch_labels:
        if label.shape[0] < max_classes:
            padding = torch.zeros((max_classes - label.shape[0], *label.shape[1:]), dtype=label.dtype)
            padded_labels.append(torch.cat((label, padding), dim=0))
        else:
            padded_labels.append(label)
    return padded_labels

# Transform function for training
def train_transforms(example_batch, image_processor, jitter, device):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [torch.tensor(x) for x in example_batch["annotation"]]
    padded_labels = pad_to_max_classes(labels)  # Pad labels to match the max size in the batch
    inputs = image_processor(images, padded_labels)

    # Padding class labels
    max_class_size = max(label.shape[0] for label in inputs["class_labels"])
    padded_class_labels = [
        F.pad(label, (0, max_class_size - label.shape[0]), "constant", 0)
        if label.shape[0] < max_class_size else label
        for label in inputs["class_labels"]
    ]

    # Padding mask labels
    max_mask_classes = max(mask.shape[0] for mask in inputs["mask_labels"])
    padded_mask_labels = [
        F.pad(mask, (0, 0, 0, 0, 0, max_mask_classes - mask.shape[0]), "constant", 0)
        if mask.shape[0] < max_mask_classes else mask
        for mask in inputs["mask_labels"]
    ]

    # Ensure tensors are on CPU before pinning
    pixel_values = torch.stack([image for image in inputs["pixel_values"]]).cpu()
    class_labels = torch.stack([label for label in padded_class_labels]).cpu()
    mask_labels = torch.stack([mask for mask in padded_mask_labels]).cpu()

    # Move tensors to GPU after they are created (if needed)
    pixel_values = pixel_values.to(device)
    class_labels = class_labels.to(device)
    mask_labels = mask_labels.to(device)

    return {
        "pixel_values": pixel_values,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }

# Transform function for validation
def val_transforms(example_batch, image_processor, jitter, device):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [torch.tensor(x) for x in example_batch["annotation"]]
    padded_labels = pad_to_max_classes(labels)
    inputs = image_processor(images, padded_labels)

    # Padding class labels
    max_class_size = max(label.shape[0] for label in inputs["class_labels"])
    padded_class_labels = [
        F.pad(label, (0, max_class_size - label.shape[0]), "constant", 0)
        if label.shape[0] < max_class_size else label
        for label in inputs["class_labels"]
    ]

    # Padding mask labels
    max_mask_classes = max(mask.shape[0] for mask in inputs["mask_labels"])
    padded_mask_labels = [
        F.pad(mask, (0, 0, 0, 0, 0, max_mask_classes - mask.shape[0]), "constant", 0)
        if mask.shape[0] < max_mask_classes else mask
        for mask in inputs["mask_labels"]
    ]

    # Ensure tensors are on CPU before pinning
    pixel_values = torch.stack([image for image in inputs["pixel_values"]]).cpu()
    class_labels = torch.stack([label for label in padded_class_labels]).cpu()
    mask_labels = torch.stack([mask for mask in padded_mask_labels]).cpu()

    # Move tensors to GPU after they are created (if needed)
    pixel_values = pixel_values.to(device)
    class_labels = class_labels.to(device)
    mask_labels = mask_labels.to(device)

    return {
        "pixel_values": pixel_values,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }




def main():
    wandb.login(key="dcbb2d83e7e9431017ffed03bf30841e0321e1b5")

    # Define device before any model-related operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse arguments
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

    # Label mapping
    label2id = {str(i): i for i in range(151)}
    id2label = {i: str(i) for i in range(151)}
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

    # Load the pre-trained model
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    # If using multiple GPUs, apply DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_args.r, 
        lora_alpha=lora_args.lora_alpha, 
        target_modules=["query", "value"],
        lora_dropout=lora_args.lora_dropout, 
        bias="lora_only", 
        modules_to_save=["decode_head"]
    )
    lora_model = get_peft_model(model, lora_config)

    # Move the model to the device (after initializing LoRA model)
    lora_model = lora_model.to(device)

    # Preprocessing datasets
    train_ds.set_transform(lambda batch: train_transforms(batch, image_processor, jitter, device))
    test_ds.set_transform(lambda batch: val_transforms(batch, image_processor, jitter, device))

    # Prepare Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,  # Deprecated warning here, but will work
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
