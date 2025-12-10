# Video Transformer Models (VideoMAE, etc.)
"""
Fine-tune video transformers for classification.
Integrated with TrainingPipeline structure.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder

# Lazy imports
def _import_torch():
    import torch
    return torch

def _import_video_libs():
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    from transformers import TrainingArguments, Trainer
    import evaluate
    return VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer, evaluate


class VideoDataset:
    """PyTorch Dataset for video classification with optimized loading."""
    
    def __init__(self, video_paths: List[str], labels: List[int], 
                 image_processor, num_frames: int = 16, sample_rate: int = 8):
        self.video_paths = video_paths
        self.labels = labels
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.sample_rate = sample_rate  # Higher = faster loading
        self._validated = False
    
    def __len__(self):
        return len(self.video_paths)
    
    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video with optimized reading."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open {video_path}")
            return []
        
        frames = []
        frame_count = 0
        max_frames = self.num_frames * self.sample_rate * 2  # Limit reading
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.sample_rate == 0:
                # Resize for faster processing
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) >= self.num_frames * 2:
                    break
            frame_count += 1
        cap.release()
        
        return frames
    
    def __getitem__(self, idx):
        torch = _import_torch()
        
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Validate first load
        if not self._validated and idx == 0:
            print(f"  Loading first video: {video_path}")
            self._validated = True
        
        frames = self._load_frames(video_path)
        
        # Sample/pad to num_frames
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame or zeros
            while len(frames) < self.num_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        inputs = self.image_processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0).float()  # Ensure float32
        
        return {"pixel_values": pixel_values, "labels": label}


class VideoTransformerModel:
    """
    Unified video transformer model compatible with TrainingPipeline.
    Supports VideoMAE with W&B monitoring and competition metrics.
    """
    
    def __init__(
        self,
        config=None,
        model_ckpt: str = "MCG-NJU/videomae-base",
        num_epochs: int = 4,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ):
        self.config = config
        self.model_ckpt = model_ckpt
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.image_processor = None
        self.trainer = None
        self.is_fitted = False
        self.output_dir = "videomae-finetuned"
        
        # Device
        self.device = config.DEVICE if config else "cuda"
    
    def _init_model(self, num_classes: int, label2id: Dict, id2label: Dict):
        """Initialize model and processor."""
        VideoMAEImageProcessor, VideoMAEForVideoClassification, _, _, _ = _import_video_libs()
        
        self.image_processor = VideoMAEImageProcessor.from_pretrained(self.model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            self.model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        self.num_frames = self.model.config.num_frames
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: List[str] = None,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        video_col: str = "video_path",
        use_wandb: bool = True,
    ) -> 'VideoTransformerModel':
        """
        Train the model. Compatible with TrainingPipeline.
        
        Args:
            X_train: Training dataframe (must contain video_col).
            y_train: Training labels.
            feature_cols: Ignored (for API compatibility).
            X_val: Validation dataframe (optional).
            y_val: Validation labels (optional).
            video_col: Column containing video paths.
            use_wandb: Enable W&B logging.
        """
        torch = _import_torch()
        _, _, TrainingArguments, Trainer, evaluate_lib = _import_video_libs()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        all_labels = list(self.label_encoder.classes_)
        label2id = {l: i for i, l in enumerate(all_labels)}
        id2label = {i: l for i, l in enumerate(all_labels)}
        
        # Initialize model
        self._init_model(len(all_labels), label2id, id2label)
        
        # Prepare datasets
        train_paths = X_train[video_col].tolist()
        train_dataset = VideoDataset(train_paths, y_encoded.tolist(), 
                                     self.image_processor, self.num_frames)
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_paths = X_val[video_col].tolist()
            val_dataset = VideoDataset(val_paths, y_val_encoded.tolist(),
                                       self.image_processor, self.num_frames)
        
        # Training arguments with W&B
        args = TrainingArguments(
            self.output_dir,
            remove_unused_columns=False,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_ratio=0.1,
            logging_steps=5,
            logging_first_step=True,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="competition_score" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            num_train_epochs=self.num_epochs,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,  # Parallel data loading
            dataloader_pin_memory=True,
            report_to="wandb" if use_wandb else "none",
            run_name=f"videomae-traffic-{self.num_epochs}ep",
        )
        
        # Competition metric: 70% Macro-F1 + 30% Accuracy
        accuracy_metric = evaluate_lib.load("accuracy")
        f1_metric = evaluate_lib.load("f1")
        
        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            labels = eval_pred.label_ids
            acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
            competition_score = 0.7 * f1 + 0.3 * acc
            return {"accuracy": acc, "macro_f1": f1, "competition_score": competition_score}
        
        def collate_fn(examples):
            pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
            labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
            return {"pixel_values": pixel_values, "labels": labels}
        
        # Trainer
        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        
        print(f"Training VideoMAE on {len(train_dataset)} videos...")
        if val_dataset:
            print(f"Validating on {len(val_dataset)} videos...")
        
        # Quick sanity check - load first sample
        print("Loading first sample to verify data pipeline...")
        try:
            sample = train_dataset[0]
            print(f"  ✓ Sample loaded: pixel_values shape = {sample['pixel_values'].shape}")
        except Exception as e:
            print(f"  ✗ Error loading sample: {e}")
            raise
        
        print("Starting training (first batch may take time)...")
        self.trainer.train()
        self.trainer.save_model()
        self.is_fitted = True
        
        if val_dataset:
            results = self.trainer.evaluate()
            print(f"\n{'='*50}")
            print("VALIDATION RESULTS")
            print(f"{'='*50}")
            print(f"Macro-F1 (70%):      {results.get('eval_macro_f1', 0):.4f}")
            print(f"Accuracy (30%):      {results.get('eval_accuracy', 0):.4f}")
            print(f"COMPETITION SCORE:   {results.get('eval_competition_score', 0):.4f}")
            print(f"{'='*50}\n")
        
        return self
    
    def predict(self, X: pd.DataFrame, video_col: str = "video_path") -> np.ndarray:
        """Predict labels (decoded strings)."""
        preds_encoded = self.predict_encoded(X, video_col)
        return self.label_encoder.inverse_transform(preds_encoded)
    
    def predict_encoded(self, X: pd.DataFrame, video_col: str = "video_path") -> np.ndarray:
        """Predict labels (encoded integers)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        torch = _import_torch()
        self.model.eval()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Filter out NaN/invalid video paths
        valid_mask = X[video_col].notna() & (X[video_col] != '')
        valid_df = X[valid_mask].copy()
        
        print(f"Predicting on {len(valid_df)}/{len(X)} valid videos...")
        
        # Predict for valid paths
        valid_predictions = []
        from tqdm import tqdm
        for video_path in tqdm(valid_df[video_col].tolist(), desc="Predicting"):
            try:
                dataset = VideoDataset([str(video_path)], [0], self.image_processor, self.num_frames)
                sample = dataset[0]
                inputs = {"pixel_values": sample["pixel_values"].unsqueeze(0).to(device)}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pred = outputs.logits.argmax(-1).item()
                valid_predictions.append(pred)
            except Exception as e:
                print(f"Error predicting {video_path}: {e}")
                valid_predictions.append(0)  # Default to first class
        
        # Build full predictions array (default=0 for invalid paths)
        predictions = np.zeros(len(X), dtype=int)
        predictions[valid_mask.values] = valid_predictions
        
        return predictions
    
    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series, video_col: str = "video_path"):
        """Evaluate with competition metric."""
        from sklearn.metrics import f1_score, accuracy_score, classification_report
        
        y_val_encoded = self.label_encoder.transform(y_val)
        y_pred_encoded = self.predict_encoded(X_val, video_col)
        
        macro_f1 = f1_score(y_val_encoded, y_pred_encoded, average='macro')
        accuracy = accuracy_score(y_val_encoded, y_pred_encoded)
        competition_score = 0.7 * macro_f1 + 0.3 * accuracy
        
        report = classification_report(y_val_encoded, y_pred_encoded, 
                                       target_names=self.label_encoder.classes_)
        
        return y_val_encoded, y_pred_encoded, {
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'competition_score': competition_score,
            'report': report
        }
    
    def get_classes(self) -> np.ndarray:
        """Get class labels."""
        return self.label_encoder.classes_
    
    def save(self, path: str) -> None:
        """Save model."""
        if self.trainer:
            self.trainer.save_model(path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config=None) -> 'VideoTransformerModel':
        """Load a saved model."""
        torch = _import_torch()
        VideoMAEImageProcessor, VideoMAEForVideoClassification, _, _, _ = _import_video_libs()
        
        instance = cls(config=config)
        instance.model = VideoMAEForVideoClassification.from_pretrained(path)
        instance.image_processor = VideoMAEImageProcessor.from_pretrained(path)
        instance.num_frames = instance.model.config.num_frames
        instance.is_fitted = True
        
        # Restore label encoder from model config
        instance.label_encoder = LabelEncoder()
        instance.label_encoder.classes_ = np.array(list(instance.model.config.id2label.values()))
        
        return instance


# Convenience function for quick training
def train_videomae(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    label_col: str = "congestion_rating",
    video_col: str = "video_path",
    output_dir: str = "videomae-traffic",
    num_epochs: int = 4,
    batch_size: int = 4,
    config=None,
    use_wandb: bool = True,
) -> VideoTransformerModel:
    """
    Convenience function to train VideoMAE from dataframes.
    """
    model = VideoTransformerModel(
        config=config,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    model.output_dir = output_dir
    
    X_train = train_df
    y_train = train_df[label_col]
    
    X_val = val_df if val_df is not None else None
    y_val = val_df[label_col] if val_df is not None else None
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, 
              video_col=video_col, use_wandb=use_wandb)
    
    return model
