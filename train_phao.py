#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

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
from transformers import (
    AutoConfig, MaskFormerImageProcessor, MaskFormerForInstanceSegmentation, get_scheduler, HfArgumentParser, TrainingArguments
)
from torchvision.transforms import ColorJitter
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import wandb

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="Xkull/phao_resize",
        metadata={"help": "Name of the dataset to use."},
    )
    train_val_split: Optional[float] = field(
        default=0.1, metadata={"help": "Percent to split off of train for validation."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/maskformer-swin-base-ade",
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

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()

    # Load dataset
    ds = load_dataset(data_args.dataset_name, trust_remote_code=True)
    ds_split = ds["train"].train_test_split(test_size=data_args.train_val_split)
    train_ds, test_ds = ds_split["train"], ds_split["test"]

    # Model and processor configuration
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    image_processor = MaskFormerImageProcessor.from_pretrained(model_args.model_name_or_path)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    # Add custom label for "phao"
    if 150 not in config.id2label:
        config.id2label[150] = "phao"
        config.label2id["phao"] = 150
    config.num_labels = len(config.id2label)

    # Prepare the model
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_args.model_name_or_path, config=config)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_args.r, lora_alpha=lora_args.lora_alpha, target_modules=["query", "value"],
        lora_dropout=lora_args.lora_dropout, bias="lora_only", modules_to_save=["decode_head"]
    )
    model = get_peft_model(model, lora_config)

    # Preprocessing datasets
    train_ds.set_transform(lambda batch: train_transforms(batch, image_processor, jitter))
    test_ds.set_transform(lambda batch: train_transforms(batch, image_processor, jitter))

    train_loader = DataLoader(train_ds, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=training_args.per_device_eval_batch_size)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    num_training_steps = len(train_loader) * training_args.num_train_epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Prepare everything with Accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # Training loop
    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            outputs = model(
                pixel_values=batch["pixel_values"],
                class_labels=batch["class_labels"],
                mask_labels=batch["mask_labels"]
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % training_args.logging_steps == 0:
                logger.info(f"Epoch {epoch} Step {step}: Loss = {loss.item()}")

    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(training_args.output_dir)

    # Compress results
    shutil.make_archive(training_args.output_dir, 'zip', training_args.output_dir)

if __name__ == "__main__":
    main()
