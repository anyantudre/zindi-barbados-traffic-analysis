# Video Transformer Models (VideoMAE, etc.)
"""
Fine-tune video transformers for classification.
Usage:
    from src.video_models import VideoMAETrainer
    trainer = VideoMAETrainer(config, num_classes=4)
    trainer.train(train_video_paths, train_labels, val_video_paths, val_labels)
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

# Lazy imports to avoid errors if not installed
def _import_video_libs():
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    from transformers import TrainingArguments, Trainer
    import evaluate
    return VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer, evaluate


class VideoDataset(Dataset):
    """Simple video dataset for classification."""
    
    def __init__(self, video_paths: List[str], labels: List[int], 
                 image_processor, num_frames: int = 16, sample_rate: int = 4):
        self.video_paths = video_paths
        self.labels = labels
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        import cv2
        
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        cap.release()
        
        # Sample/pad to num_frames
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad by repeating last frame
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Process frames - image_processor expects list of frames
        inputs = self.image_processor(list(frames), return_tensors="pt")
        # Shape: (1, num_frames, channels, height, width) -> remove batch dim
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {"pixel_values": pixel_values, "labels": label}


class VideoMAETrainer:
    """Fine-tune VideoMAE for video classification."""
    
    def __init__(
        self,
        num_classes: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        model_ckpt: str = "MCG-NJU/videomae-base",
        output_dir: str = "videomae-finetuned",
        num_epochs: int = 4,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ):
        self.model_ckpt = model_ckpt
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.label2id = label2id
        self.id2label = id2label
        
        # Load model and processor
        VideoMAEImageProcessor, VideoMAEForVideoClassification, _, _, _ = _import_video_libs()
        
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        self.num_frames = self.model.config.num_frames
    
    def train(
        self,
        train_video_paths: List[str],
        train_labels: List[int],
        val_video_paths: List[str] = None,
        val_labels: List[int] = None,
    ):
        """Fine-tune the model."""
        _, _, TrainingArguments, Trainer, evaluate_lib = _import_video_libs()
        
        # Create datasets
        train_dataset = VideoDataset(
            train_video_paths, train_labels, 
            self.image_processor, self.num_frames
        )
        
        val_dataset = None
        if val_video_paths and val_labels:
            val_dataset = VideoDataset(
                val_video_paths, val_labels,
                self.image_processor, self.num_frames
            )
        
        # Training arguments
        args = TrainingArguments(
            self.output_dir,
            remove_unused_columns=False,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="accuracy" if val_dataset else None,
            num_train_epochs=self.num_epochs,
            fp16=torch.cuda.is_available(),
        )
        
        # Metrics
        metric = evaluate_lib.load("accuracy")
        
        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=predictions, references=eval_pred.label_ids)
        
        def collate_fn(examples):
            # Stack pixel_values: each is (num_frames, C, H, W)
            pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
            labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
            return {"pixel_values": pixel_values, "labels": labels}
        
        # Trainer
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        
        print(f"Training VideoMAE on {len(train_dataset)} videos...")
        trainer.train()
        trainer.save_model()
        
        if val_dataset:
            results = trainer.evaluate()
            print(f"Validation accuracy: {results.get('eval_accuracy', 'N/A')}")
        
        return trainer
    
    def predict(self, video_paths: List[str]) -> np.ndarray:
        """Run inference on videos."""
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        predictions = []
        
        for video_path in video_paths:
            dataset = VideoDataset([video_path], [0], self.image_processor, self.num_frames)
            sample = dataset[0]
            
            inputs = {"pixel_values": sample["pixel_values"].unsqueeze(0).to(device)}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = outputs.logits.argmax(-1).item()
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_labels(self, video_paths: List[str]) -> List[str]:
        """Run inference and return label names."""
        preds = self.predict(video_paths)
        return [self.id2label[p] for p in preds]


def train_videomae(
    train_df,
    val_df,
    label_col: str = "congestion_rating",
    video_col: str = "video_path",
    output_dir: str = "videomae-traffic",
    num_epochs: int = 4,
    batch_size: int = 4,
) -> VideoMAETrainer:
    """
    Convenience function to train VideoMAE from dataframes.
    
    Args:
        train_df: Training DataFrame with video_path and label columns.
        val_df: Validation DataFrame.
        label_col: Name of label column.
        video_col: Name of video path column.
        output_dir: Directory to save model.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        
    Returns:
        Trained VideoMAETrainer instance.
    """
    # Get unique labels
    all_labels = sorted(train_df[label_col].unique())
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Prepare data
    train_paths = train_df[video_col].tolist()
    train_labels = [label2id[l] for l in train_df[label_col]]
    
    val_paths = val_df[video_col].tolist()
    val_labels = [label2id[l] for l in val_df[label_col]]
    
    # Train
    trainer = VideoMAETrainer(
        num_classes=len(all_labels),
        label2id=label2id,
        id2label=id2label,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    
    trainer.train(train_paths, train_labels, val_paths, val_labels)
    
    return trainer

