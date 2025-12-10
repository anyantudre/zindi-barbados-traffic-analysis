# Video feature extraction
"""
Unified feature extraction module with consistent interface for all extractors.
Each extractor follows the same pattern: extract(video_path) -> Dict
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List


# =============================================================================
# Base Extractors (CPU-only, lightweight)
# =============================================================================

class SimpleExtractor:
    """Extract simple statistical features (brightness, colors)."""
    
    def __init__(self, sample_interval: int = 30, target_size: int = 224):
        self.sample_interval = sample_interval
        self.target_size = target_size
    
    def extract(self, video_path: str) -> Dict:
        """Extract simple features from a single video."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        sample_count = 0
        
        mean = np.zeros(3, dtype=np.float64)
        M2 = np.zeros(3, dtype=np.float64)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_interval == 0:
                if self.target_size:
                    frame = cv2.resize(frame, (self.target_size, self.target_size))
                
                sample_mean = frame.mean(axis=(0, 1))
                sample_count += 1
                delta = sample_mean - mean
                mean += delta / sample_count
                M2 += delta * (sample_mean - mean)
            
            frame_idx += 1
        
        cap.release()
        
        if sample_count == 0:
            return None
        
        std = np.sqrt(M2 / sample_count)
        
        return {
            "video_path": video_path,
            "num_frames_sampled": sample_count,
            "mean_r": mean[2], "mean_g": mean[1], "mean_b": mean[0],
            "std_r": std[2], "std_g": std[1], "std_b": std[0],
            "brightness": mean.mean(),
        }


class FlowExtractor:
    """Extract optical flow features using Farneback."""
    
    def __init__(self, sample_interval: int = 15):
        self.sample_interval = sample_interval
    
    def extract(self, video_path: str) -> Dict:
        """Extract optical flow features from a single video."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        prev_gray = None
        flow_mags, flow_angles = [], []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_mags.append(mag.mean())
                    flow_angles.append(ang.mean())
                prev_gray = gray
            
            frame_idx += 1
        
        cap.release()
        
        return {
            "video_path": video_path,
            "flow_mean": np.mean(flow_mags) if flow_mags else 0,
            "flow_std": np.std(flow_mags) if flow_mags else 0,
            "flow_max": np.max(flow_mags) if flow_mags else 0,
            "flow_angle_mean": np.mean(flow_angles) if flow_angles else 0,
            "flow_angle_std": np.std(flow_angles) if flow_angles else 0,
        }


# =============================================================================
# GPU Extractors (require ultralytics/torch)
# =============================================================================

class YOLOExtractor:
    """Extract vehicle detection features using YOLOv8 native video streaming."""
    
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
    
    def __init__(self, sample_interval: int = 15, device: str = "cuda"):
        self.sample_interval = sample_interval
        self.device = device
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
    
    def extract(self, video_path: str) -> Dict:
        """Extract YOLO features using native video streaming (GPU optimized)."""
        self._load_model()
        
        vehicle_counts, bbox_areas = [], []
        vehicle_types = {k: 0 for k in self.VEHICLE_CLASSES}
        
        # Use YOLO's native video streaming - much faster
        results = self.model.predict(
            source=video_path,
            stream=True,
            verbose=False,
            vid_stride=self.sample_interval  # Skip frames natively
        )
        
        for result in results:
            count = 0
            for box in result.boxes:
                name = result.names[int(box.cls)]
                if name in self.VEHICLE_CLASSES:
                    count += 1
                    vehicle_types[name] += 1
                    xyxy = box.xyxy.cpu().numpy()[0]
                    bbox_areas.append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            vehicle_counts.append(count)
        
        if not vehicle_counts:
            return {"video_path": video_path, "vehicle_count_mean": 0, "vehicle_count_std": 0,
                    "vehicle_count_max": 0, "avg_bbox_area": 0,
                    **{f"{k}_count": 0 for k in self.VEHICLE_CLASSES}}
        
        return {
            "video_path": video_path,
            "vehicle_count_mean": np.mean(vehicle_counts),
            "vehicle_count_std": np.std(vehicle_counts),
            "vehicle_count_max": np.max(vehicle_counts),
            "avg_bbox_area": np.mean(bbox_areas) if bbox_areas else 0,
            **{f"{k}_count": v for k, v in vehicle_types.items()}
        }


class EmbeddingExtractor:
    """Extract scene embeddings using EfficientNet-B0 with batching."""
    
    def __init__(self, sample_interval: int = 15, device: str = "cuda", 
                 n_components: int = 64, batch_size: int = 32):
        self.sample_interval = sample_interval
        self.device = device
        self.n_components = n_components
        self.batch_size = batch_size
        self.model = None
        self.preprocess = None
    
    def _load_model(self):
        if self.model is None:
            import torch
            from torchvision import models, transforms
            
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.model.to(self.device).eval()
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def extract(self, video_path: str) -> Dict:
        """Extract embedding features using batch inference."""
        import torch
        self._load_model()
        
        # Collect frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        cap.release()
        
        n = self.n_components
        if not frames:
            return {"video_path": video_path,
                    **{f"emb_mean_{i}": 0.0 for i in range(n)},
                    **{f"emb_std_{i}": 0.0 for i in range(n)}}
        
        # Batch inference
        all_embs = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            batch_tensor = torch.stack([self.preprocess(f) for f in batch]).to(self.device)
            
            with torch.no_grad():
                out = self.model.features(batch_tensor)
                out = self.model.avgpool(out)
                embs = out.squeeze(-1).squeeze(-1).cpu().numpy()
            all_embs.append(embs)
        
        emb = np.vstack(all_embs)[:, :n]
        return {
            "video_path": video_path,
            **{f"emb_mean_{i}": emb[:, i].mean() for i in range(n)},
            **{f"emb_std_{i}": emb[:, i].std() for i in range(n)}
        }


class VideoMAEExtractor:
    """Extract video embeddings using VideoMAE transformer with batch inference."""
    
    def __init__(self, device: str = "cuda", n_components: int = 64, 
                 num_frames: int = 16, batch_size: int = 16,
                 model_ckpt: str = "MCG-NJU/videomae-base"):
        self.device = device
        self.n_components = n_components
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.model_ckpt = model_ckpt
        self.model = None
        self.processor = None
    
    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import VideoMAEImageProcessor, VideoMAEModel
            
            self.processor = VideoMAEImageProcessor.from_pretrained(self.model_ckpt)
            self.model = VideoMAEModel.from_pretrained(
                self.model_ckpt,
                attn_implementation="sdpa",
                torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            print(f"VideoMAE model loaded on {self.device}")
    
    def _read_frames(self, video_path: str) -> List[np.ndarray]:
        """Read and uniformly sample frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.num_frames]
    
    def _make_empty_result(self, video_path: str) -> Dict:
        """Create empty result for failed videos."""
        return {"video_path": video_path, **{f"vmae_{i}": 0.0 for i in range(self.n_components)}}
    
    def extract_batch(self, video_paths: List[str]) -> List[Dict]:
        """Extract features from multiple videos using batch inference."""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self._load_model()
        
        n = self.n_components
        results = []
        
        # Read frames in parallel using threads (I/O bound)
        valid_paths = [(i, p) for i, p in enumerate(video_paths) if os.path.exists(p)]
        
        def read_with_index(args):
            i, path = args
            frames = self._read_frames(path)
            return (i, frames) if frames else None
        
        all_frames = []
        valid_indices = []
        
        with ThreadPoolExecutor(max_workers=min(8, len(valid_paths))) as executor:
            results_iter = executor.map(read_with_index, valid_paths)
            for result in results_iter:
                if result:
                    i, frames = result
                    all_frames.append(frames)
                    valid_indices.append(i)
        
        if not all_frames:
            return [self._make_empty_result(p) for p in video_paths]
        
        # Process in batches
        all_embeddings = []
        for batch_start in range(0, len(all_frames), self.batch_size):
            batch_frames = all_frames[batch_start:batch_start + self.batch_size]
            
            # Process each video's frames and stack into batch
            batch_inputs = [self.processor(frames, return_tensors="pt")["pixel_values"] 
                           for frames in batch_frames]
            batch_tensor = torch.cat(batch_inputs, dim=0).to(self.device, dtype=torch.float16)
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                outputs = self.model(pixel_values=batch_tensor)
                cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
                mean_emb = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
            
            # Combine and truncate
            for cls, mean in zip(cls_emb, mean_emb):
                combined = np.concatenate([cls, mean])[:n]
                if len(combined) < n:
                    combined = np.pad(combined, (0, n - len(combined)))
                all_embeddings.append(combined)
        
        # Build results for all paths
        emb_idx = 0
        for i, path in enumerate(video_paths):
            if i in valid_indices:
                emb = all_embeddings[emb_idx]
                results.append({"video_path": path, **{f"vmae_{j}": float(emb[j]) for j in range(n)}})
                emb_idx += 1
            else:
                results.append(self._make_empty_result(path))
        
        return results
    
    def extract(self, video_path: str) -> Dict:
        """Extract VideoMAE embeddings for single video."""
        return self.extract_batch([video_path])[0]


# =============================================================================
# Unified extraction interface
# =============================================================================

def extract_features(extractor, video_paths: List[str], cache_path: str, 
                     extract: bool = False) -> pd.DataFrame:
    """
    Extract or load features using the given extractor.
    
    Args:
        extractor: Feature extractor instance with extract(video_path) method.
        video_paths: List of video file paths.
        cache_path: Path to CSV file (local or Kaggle path).
        extract: If True, extract features. If False, load from cache_path.
        
    Returns:
        DataFrame with extracted features.
    """
    if not extract:
        print(f"Loading features from {cache_path}")
        return pd.read_csv(cache_path)
    
    print(f"Extracting {extractor.__class__.__name__} features...")
    
    # Use batch extraction if available (VideoMAEExtractor)
    if hasattr(extractor, 'extract_batch'):
        batch_size = getattr(extractor, 'batch_size', 16)
        records = []
        valid_paths = [p for p in video_paths if os.path.exists(p)]
        
        for i in tqdm(range(0, len(valid_paths), batch_size), 
                      desc=f"{extractor.__class__.__name__} (batch={batch_size})"):
            batch_paths = valid_paths[i:i + batch_size]
            try:
                batch_results = extractor.extract_batch(batch_paths)
                records.extend([r for r in batch_results if r])
            except Exception as e:
                print(f"Batch error: {e}")
                # Fallback to single extraction
                for path in batch_paths:
                    try:
                        feat = extractor.extract(path)
                        if feat:
                            records.append(feat)
                    except Exception as e2:
                        print(f"Error: {path}: {e2}")
    else:
        # Single video extraction (other extractors)
        records = []
        for video_path in tqdm(video_paths, desc=extractor.__class__.__name__):
            if os.path.exists(video_path):
                try:
                    feat = extractor.extract(video_path)
                    if feat:
                        records.append(feat)
                except Exception as e:
                    print(f"Error: {video_path}: {e}")
    
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} records to {cache_path}")
    return df


def get_all_features(config, video_paths: List[str] = None) -> pd.DataFrame:
    """
    Get all enabled features (simple + semantic) with unified interface.
    
    Args:
        config: Configuration object with feature flags and paths.
        video_paths: List of video paths.
        
    Returns:
        DataFrame with all combined features.
    """
    features = None
    
    # Feature types configuration: (flag, path, extractor_factory)
    feature_types = [
        ('EXTRACT_SIMPLE', 'SIMPLE_FEATURES_PATH', 
         lambda: SimpleExtractor(config.SAMPLE_INTERVAL, config.TARGET_SIZE)),
        ('EXTRACT_YOLO', 'YOLO_FEATURES_PATH',
         lambda: YOLOExtractor(config.SAMPLE_INTERVAL, config.DEVICE)),
        ('EXTRACT_FLOW', 'FLOW_FEATURES_PATH',
         lambda: FlowExtractor(config.SAMPLE_INTERVAL)),
        ('EXTRACT_EMBEDDINGS', 'EMBEDDING_FEATURES_PATH',
         lambda: EmbeddingExtractor(config.SAMPLE_INTERVAL, config.DEVICE, config.EMBEDDING_DIM)),
        ('EXTRACT_VIDEOMAE', 'VIDEOMAE_FEATURES_PATH',
         lambda: VideoMAEExtractor(config.DEVICE, config.VIDEOMAE_DIM, config.VIDEOMAE_NUM_FRAMES)),
    ]
    
    for flag_name, path_name, make_extractor in feature_types:
        extract_flag = getattr(config, flag_name, False)
        cache_path = getattr(config, path_name, None)
        
        if cache_path is None:
            continue
        
        # Check if file exists when not extracting
        if not extract_flag and not os.path.exists(cache_path):
            print(f"Warning: {cache_path} not found, skipping {flag_name}")
            continue
        
        extractor = make_extractor()
        df = extract_features(extractor, video_paths or [], cache_path, extract_flag)
        
        # Add video_id for robust merging (filename without path)
        df['video_id'] = df['video_path'].apply(lambda x: os.path.basename(x))
        
        if features is None:
            features = df
        else:
            # Merge on video_id (filename) instead of full path
            df_to_merge = df.drop(columns=['video_path'])
            features = features.merge(df_to_merge, on='video_id', how='left')
        
        print(f"Loaded {path_name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Remove helper column
    if features is not None and 'video_id' in features.columns:
        features = features.drop(columns=['video_id'])
    
    return features
