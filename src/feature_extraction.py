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
from PIL import Image

import torch
from concurrent.futures import ThreadPoolExecutor


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
        ('EXTRACT_PE', 'PE_FEATURES_PATH',
        lambda: PerceptionEncoderExtractor(config.DEVICE, config.PE_NUM_FRAMES, config.PE_DIM)),
        ('EXTRACT_YOLOFLOW', 'YOLOFLOW_FEATURES_PATH',
         lambda: YOLOFlowExtractor(config.SAMPLE_INTERVAL, config.DEVICE)),
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


def _read_frames_pyav(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """Read frames using PyAV (10-20x faster than OpenCV for seeking)."""
    try:
        import av
    except ImportError:
        # Fallback to OpenCV if PyAV not available
        print("PyAV not available, falling back to OpenCV")
        return _read_frames_cv2(video_path, num_frames)
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames or int(stream.duration * stream.average_rate)
        
        if total_frames <= 0:
            container.close()
            return []
        
        indices = set(np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int))
        frames = []
        
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_rgb().to_ndarray())
            if len(frames) >= num_frames:
                break
        
        container.close()
        
        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:num_frames]
    except Exception:
        print("Error reading frames with PyAV, falling back to OpenCV")
        return []


def _read_frames_cv2(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """Fallback OpenCV reader."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        print("Error reading frames with OpenCV")
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    return frames[:num_frames]


# =============================================================================
# Base Extractors (CPU-only, lightweight)
# =============================================================================

# class SimpleExtractor:
#     """Fast, minimalist statistical features."""

#     def __init__(self, sample_interval=30, target_size=224):
#         self.sample_interval = sample_interval
#         self.target_size = target_size

#     def extract(self, video_path: str):
#         cap = cv2.VideoCapture(video_path)
#         frame_idx = 0
#         prev_frame = None

#         # Running stats
#         brightness_vals = []
#         diff_vals = []
#         sat_vals = []
#         edge_vals = []
#         blur_vals = []
#         entropy_vals = []
#         dark_vals = []

#         sample_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_idx % self.sample_interval == 0:
#                 if self.target_size:
#                     frame = cv2.resize(frame, (self.target_size, self.target_size))

#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#                 # 1. brightness
#                 brightness_vals.append(gray.mean())

#                 # 2. motion diff
#                 if prev_frame is not None:
#                     diff_vals.append(np.mean(np.abs(gray - prev_frame)))

#                 # 3. color saturation
#                 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#                 sat_vals.append(hsv[:, :, 1].mean())

#                 # 4. edges
#                 edges = cv2.Canny(gray, 100, 200)
#                 edge_vals.append(edges.mean())

#                 # 5. blurriness
#                 blur_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())

#                 # 6. entropy
#                 hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
#                 entropy_vals.append(-(hist * np.log2(hist + 1e-9)).sum())

#                 # 7. dark channel (haze / shadow score)
#                 dark_vals.append(np.min(frame, axis=2).mean())

#                 prev_frame = gray.copy()
#                 sample_count += 1

#             frame_idx += 1

#         cap.release()
#         if sample_count == 0:
#             return None

#         return {
#             "video_path": video_path,
#             "num_frames_sampled": sample_count,

#             "brightness_mean": float(np.mean(brightness_vals)),
#             "brightness_std": float(np.std(brightness_vals)),

#             "motion_mean": float(np.mean(diff_vals)) if diff_vals else 0,
#             "motion_std": float(np.std(diff_vals)) if diff_vals else 0,

#             "sat_mean": float(np.mean(sat_vals)),
#             "sat_std": float(np.std(sat_vals)),

#             "edge_density": float(np.mean(edge_vals)),
#             "blur_mean": float(np.mean(blur_vals)),
#             "entropy_mean": float(np.mean(entropy_vals)),
#             "dark_channel_mean": float(np.mean(dark_vals)),
#         }

class SimpleExtractor:
    """
    Extremely lightweight feature extractor for roundabout traffic videos.
    All operations run on tiny grayscale frames with minimal computation.
    Each block is optional.
    """

    def __init__(
        self,
        sample_interval=10,
        tiny_size=64,

        use_intensity_motion=True,
        use_edge_activity=True,
        use_directionality=True,
        use_brightness=True,
        use_color_dominance=True,
        use_scene_occupancy=True,
        use_blob_count=True,
        use_frame_diff_var=True,
    ):
        self.sample_interval = sample_interval
        self.tiny_size = tiny_size

        # toggles
        self.use_intensity_motion = use_intensity_motion
        self.use_edge_activity = use_edge_activity
        self.use_directionality = use_directionality
        self.use_brightness = use_brightness
        self.use_color_dominance = use_color_dominance
        self.use_scene_occupancy = use_scene_occupancy
        self.use_blob_count = use_blob_count
        self.use_frame_diff_var = use_frame_diff_var

    def extract(self, video_path: str):
        import av
        container = av.open(video_path)

        prev_gray = None
        prev_small = None

        # storage
        motion = []
        edges = []
        mag_vals = []
        ang_vals = []
        brightness = []
        colors = []
        occupancy = []
        blob_counts = []
        diff_vars = []

        frame_idx = 0

        for frame in container.decode(video=0):

            if frame_idx % self.sample_interval != 0:
                frame_idx += 1
                continue

            img = frame.to_ndarray(format="bgr24")

            # ultra-fast: precompute grayscale tiny frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tiny = cv2.resize(gray, (self.tiny_size, self.tiny_size))

            # 1) intensity motion (difference)
            if self.use_intensity_motion and prev_gray is not None:
                motion.append(cv2.absdiff(tiny, prev_small).mean())

            # 2) edge activity (tiny)
            if self.use_edge_activity:
                e = cv2.Canny(tiny, 30, 90)
                edges.append(e.mean())

            # 3) directionality (super tiny flow)
            if self.use_directionality and prev_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, tiny, None,
                    pyr_scale=0.5, levels=1,
                    winsize=9, iterations=1,
                    poly_n=5, poly_sigma=1.1, flags=0,
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag_vals.append(mag.mean())
                ang_vals.append(ang.mean())

            # 4) brightness
            if self.use_brightness:
                brightness.append(tiny.mean())

            # 5) color dominance (only sample 32x32 for RGB)
            if self.use_color_dominance:
                small_rgb = cv2.resize(img, (32, 32))
                cm = small_rgb.mean(axis=(0, 1))
                colors.append(cm)

            # 6) occupancy (binary threshold)
            if self.use_scene_occupancy:
                th = tiny.mean()
                occupancy.append(float((tiny > th).mean()))

            # 7) blob count (fast 32x32)
            if self.use_blob_count:
                small32 = cv2.resize(tiny, (32, 32))
                _, bin_ = cv2.threshold(small32, small32.mean(), 255, cv2.THRESH_BINARY)
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(bin_)
                blob_counts.append(num_labels)

            # 8) frame diff variance (variation measure)
            if self.use_frame_diff_var and prev_small is not None:
                diff = cv2.absdiff(tiny, prev_small)
                diff_vars.append(diff.var())

            prev_gray = gray
            prev_small = tiny
            frame_idx += 1

        container.close()

        out = {"video_path": video_path}

        # finalize
        if self.use_intensity_motion:
            out["motion_mean"] = float(np.mean(motion)) if motion else 0.0

        if self.use_edge_activity:
            out["edge_mean"] = float(np.mean(edges)) if edges else 0.0

        if self.use_directionality:
            out["flow_mag_mean"] = float(np.mean(mag_vals)) if mag_vals else 0.0
            out["flow_ang_mean"] = float(np.mean(ang_vals)) if ang_vals else 0.0

        if self.use_brightness:
            out["brightness_mean"] = float(np.mean(brightness)) if brightness else 0.0
            out["brightness_std"] = float(np.std(brightness)) if brightness else 0.0

        if self.use_color_dominance:
            if colors:
                cm = np.mean(colors, axis=0)
                out["color_r"], out["color_g"], out["color_b"] = cm[2], cm[1], cm[0]
            else:
                out["color_r"] = out["color_g"] = out["color_b"] = 0.0

        if self.use_scene_occupancy:
            out["occupancy_mean"] = float(np.mean(occupancy)) if occupancy else 0.0

        if self.use_blob_count:
            out["blob_mean"] = float(np.mean(blob_counts)) if blob_counts else 0.0

        if self.use_frame_diff_var:
            out["diff_var_mean"] = float(np.mean(diff_vars)) if diff_vars else 0.0

        return out


class FlowExtractor:
    """Extract optical flow features using Farneback, optimized with PyAV and reduced resolution."""

    def __init__(self, sample_interval: int = 15, flow_resize: int = 128):
        self.sample_interval = sample_interval
        self.flow_resize = flow_resize  # internal resolution for faster flow

    def extract(self, video_path: str) -> Dict:
        """Extract optical flow features from a single video."""
        import av
        container = av.open(video_path)

        prev_gray = None
        flow_mags = []
        flow_angles = []

        frame_idx = 0

        # Pre-allocated Farneback buffers
        fb_params = dict(
            pyr_scale=0.5, 
            levels=3, 
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        for frame in container.decode(video=0):
            if frame_idx % self.sample_interval != 0:
                frame_idx += 1
                continue

            # PyAV gives ndarray directly, faster than cv2.VideoCapture
            img = frame.to_ndarray(format="bgr24")

            # grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # reduced resolution inside the algorithm (faster flow)
            if self.flow_resize is not None:
                gray_small = cv2.resize(gray, (self.flow_resize, self.flow_resize))
            else:
                gray_small = gray

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray_small, None, **fb_params
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mags.append(mag.mean())
                flow_angles.append(ang.mean())

            prev_gray = gray_small
            frame_idx += 1

        container.close()

        return {
            "video_path": video_path,
            "flow_mean": float(np.mean(flow_mags)) if flow_mags else 0.0,
            "flow_std": float(np.std(flow_mags)) if flow_mags else 0.0,
            "flow_max": float(np.max(flow_mags)) if flow_mags else 0.0,
            "flow_angle_mean": float(np.mean(flow_angles)) if flow_angles else 0.0,
            "flow_angle_std": float(np.std(flow_angles)) if flow_angles else 0.0,
        }


# =============================================================================
# GPU Extractors (require ultralytics/torch)
# =============================================================================

class YOLOExtractor:
    """Extract dynamic object detection features using YOLOv11 native video streaming."""

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, sample_interval: int = 15, device: str = "cuda", imgsz: int = 640):
        self.sample_interval = sample_interval
        self.device = device
        self.model = None
        self.imgsz = imgsz

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n.pt")
            self.model.to(self.device)

    def extract(self, video_path: str) -> dict:
        """Extract YOLO features using native video streaming (GPU optimized)."""
        import numpy as np
        from scipy.stats import linregress
        import os

        self._load_model()

        class_counts = {}
        vehicle_counts = {v: 0 for v in self.VEHICLE_CLASSES}
        frame_object_counts = []
        bbox_areas = []
        bbox_aspect_ratios = []

        results = self.model.predict(
            source=video_path,
            stream=True,
            verbose=False,
            vid_stride=self.sample_interval,
            imgsz=self.imgsz
        )

        for result in results:
            frame_count = 0
            frame_bbox_areas = []

            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]

                # Dynamic class counts
                key = f"class_count_{cls_name}"
                class_counts[key] = class_counts.get(key, 0) + 1

                # Vehicle counts
                if cls_name in self.VEHICLE_CLASSES:
                    vehicle_counts[cls_name] += 1

                # bbox area & aspect ratio
                xyxy = box.xyxy.cpu().numpy()[0]
                width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                area = width * height
                frame_bbox_areas.append(area)
                bbox_areas.append(area)
                if height > 0:
                    bbox_aspect_ratios.append(width / height)
                frame_count += 1

            frame_object_counts.append(frame_count)

        # Aggregate features
        total_objects = sum(frame_object_counts)
        total_bbox_area = sum(bbox_areas)

        # Traffic growth metrics
        first_count = frame_object_counts[0] if frame_object_counts else 0
        last_count = frame_object_counts[-1] if frame_object_counts else 0
        traffic_growth_ratio = (last_count - first_count) / max(1, first_count)

        # Linear regression slope for per-frame counts
        # if len(frame_object_counts) > 1:
        #     slope, _, _, _, _ = linregress(range(len(frame_object_counts)), frame_object_counts)
        # else:
        #     slope = 0.0

        # traffic_percentile_change = np.percentile(frame_object_counts, 90) - np.percentile(frame_object_counts, 10) \
        #     if frame_object_counts else 0.0

        result_dict = {
            "video_path": video_path,
            "total_objects": total_objects,
            "object_count_mean": float(np.mean(frame_object_counts)) if frame_object_counts else 0.0,
            "object_count_std": float(np.std(frame_object_counts)) if frame_object_counts else 0.0,
            "object_count_max": int(np.max(frame_object_counts)) if frame_object_counts else 0,
            "avg_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0.0,
            "total_bbox_area": float(total_bbox_area),
            "avg_bbox_aspect_ratio": float(np.mean(bbox_aspect_ratios)) if bbox_aspect_ratios else 0.0,
            "object_count_evolution": frame_object_counts,
            "traffic_growth_ratio": float(traffic_growth_ratio),
            # "traffic_slope": float(slope),
            # "traffic_percentile_change": float(traffic_percentile_change),
            **class_counts,
            **vehicle_counts
        }

        # Print one line per video
        vid_name = os.path.basename(video_path)
        print(f"[YOLOExtractor] {vid_name} → {result_dict}")

        return result_dict



class YOLOFlowExtractor:
    """
    Traffic-flow feature extractor using YOLO11 tracking with persistence.
    """

    def __init__(
        self,
        sample_interval: int = 5,
        mask_interval: int = 5,
        imgsz: int = 640,
        conf: float = 0.25,
        device: str = "cuda",
    ):
        self.sample_interval = sample_interval
        self.mask_interval = mask_interval
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.model = None

        self.zones = {
            "zone_1": (0.0, 0.33),
            "zone_2": (0.33, 0.66),
            "zone_3": (0.66, 1.0),
        }

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n-seg.pt")
            self.model.to(self.device)

    def _get_zone(self, cx_norm):
        for zname, (low, high) in self.zones.items():
            if low <= cx_norm < high:
                return zname
        return "zone_3"

    def extract(self, video_path: str, mode: str = "deep") -> Dict:
        import numpy as np
        import os

        self._load_model()
        compute_masks = mode == "deep"
        compute_speed = mode == "deep"

        frame_idx = 0
        prev_positions = {}
        speed_dists = []
        stopped_counts = 0
        cluster_sizes = []
        coverage_pixels = []

        zone_counts = {z: 0 for z in self.zones}

        # Dynamic class counts
        class_counts = {}   # filled dynamically once we get result.names

        results = self.model.track(
            source=video_path,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            vid_stride=self.sample_interval,
            device=self.device,
            stream=True,
            verbose=False
        )

        for result in results:
            if not class_counts:
                # Initialize class counters dynamically
                class_counts = {f"class_count_{name}": 0 for name in result.names.values()}

            frame = result.orig_img
            H, W = frame.shape[:2]
            ids_this_frame = []
            positions = {}

            # Mask coverage
            if compute_masks and result.masks is not None and (frame_idx % self.mask_interval == 0):
                m = result.masks.data
                coverage_pixels.append(float(m.sum()) / (H * W))

            if result.boxes is not None:
                # Get tracking IDs if available
                try:
                    track_ids = result.boxes.id.int().cpu().tolist()
                except Exception:
                    track_ids = [None] * len(result.boxes)

                for box, tid in zip(result.boxes, track_ids):
                    cls_id = int(box.cls)
                    cls_name = result.names[cls_id]

                    # count per class
                    class_counts[f"class_count_{cls_name}"] += 1

                    # basic tracking-based features
                    xyxy = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    cx_norm = cx / W

                    ids_this_frame.append(tid)
                    if tid is not None:
                        positions[tid] = (cx, cy)

                    zone_counts[self._get_zone(cx_norm)] += 1

            # speed / stopped metrics
            if compute_speed:
                for tid, pos in positions.items():
                    if tid in prev_positions:
                        px, py = prev_positions[tid]
                        cx, cy = pos
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        speed_dists.append(dist)
                        if dist < 0.5:
                            stopped_counts += 1
                prev_positions.update(positions)

            cluster_sizes.append(len(ids_this_frame))
            frame_idx += 1

        # Output dict
        result_dict = {
            "video_path": video_path,
            "cluster_size_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "cluster_size_max": float(np.max(cluster_sizes)) if cluster_sizes else 0.0,
            **{f"{z}_count": int(c) for z, c in zone_counts.items()},
            **class_counts,
        }

        if compute_speed:
            result_dict.update({
                "speed_mean": float(np.mean(speed_dists)) if speed_dists else 0.0,
                "speed_std": float(np.std(speed_dists)) if speed_dists else 0.0,
                "stopped_ratio": float(stopped_counts / max(1, sum(cluster_sizes))),
            })

        if compute_masks:
            result_dict.update({
                "mask_coverage_mean": float(np.mean(coverage_pixels)) if coverage_pixels else 0.0,
            })

        # Clean one-line print
        vid_name = os.path.basename(video_path)
        print(f"[YOLOFlowExtractor] {vid_name} → {result_dict}")

        return result_dict




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
    """Extract video embeddings using VideoMAE transformer with optimized batch inference."""
    
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
        """Read frames using PyAV (fast) with OpenCV fallback."""
        return _read_frames_pyav(video_path, self.num_frames)
    
    def _make_empty_result(self, video_path: str) -> Dict:
        """Create empty result for failed videos."""
        return {"video_path": video_path, **{f"vmae_{i}": 0.0 for i in range(self.n_components)}}
    
    def extract_batch(self, video_paths: List[str]) -> List[Dict]:
        """Extract features from multiple videos using optimized batch inference."""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self._load_model()
        
        n = self.n_components
        
        # Read frames in parallel using threads
        valid_paths = [(i, p) for i, p in enumerate(video_paths) if os.path.exists(p)]
        
        def read_task(args):
            i, path = args
            frames = self._read_frames(path)
            return (i, frames) if frames else None
        
        all_frames = []
        valid_indices = []
        
        with ThreadPoolExecutor(max_workers=min(8, len(valid_paths) or 1)) as executor:
            for result in executor.map(read_task, valid_paths):
                if result:
                    i, frames = result
                    all_frames.append(frames)
                    valid_indices.append(i)
        
        if not all_frames:
            return [self._make_empty_result(p) for p in video_paths]
        
        # Process each video's frames separately (processor expects frames from ONE video)
        batch_inputs = [self.processor(frames, return_tensors="pt")["pixel_values"] 
                       for frames in all_frames]
        batch_tensor = torch.cat(batch_inputs, dim=0).to(self.device, dtype=torch.float16)
        
        # GPU inference
        with torch.no_grad():
            outputs = self.model(pixel_values=batch_tensor)
            cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            mean_emb = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
        
        # Combine and build results
        results = []
        emb_idx = 0
        for i, path in enumerate(video_paths):
            if i in valid_indices:
                combined = np.concatenate([cls_emb[emb_idx], mean_emb[emb_idx]])[:n]
                if len(combined) < n:
                    combined = np.pad(combined, (0, n - len(combined)))
                results.append({"video_path": path, **{f"vmae_{j}": float(combined[j]) for j in range(n)}})
                emb_idx += 1
            else:
                results.append(self._make_empty_result(path))
        
        return results
    
    def extract(self, video_path: str) -> Dict:
        """Extract VideoMAE embeddings for single video."""
        return self.extract_batch([video_path])[0]


class PerceptionEncoderExtractor:
    """
    Super simple video embedding extractor using Meta PE.
    One video at a time. No batching, no threads.
    Decoding happens on CPU; only embeddings run on GPU.
    """

    def __init__(self,
                 model_ckpt="PE-Core-L14-336",
                 num_frames=8,
                 n_components=1024,
                 device="cuda"):

        self.model_ckpt = model_ckpt
        self.num_frames = num_frames
        self.n_components = n_components
        self.device = device

        self.model = None
        self.preprocess = None

    def _load_model(self):
        if self.model is None:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms

            # Load PE model and preprocessing transform
            self.model = pe.CLIP.from_config(self.model_ckpt, pretrained=True)
            self.model = self.model.to(self.device).eval()
            self.preprocess = transforms.get_image_transform(self.model.image_size)

    def _read_frames(self, video_path: str) -> List[np.ndarray]:
        """Read frames using PyAV (fast) with OpenCV fallback."""
        return _read_frames_pyav(video_path, self.num_frames)

    def extract(self, video_path):
        """
        Extract video embedding for a single video.
        Returns dict: {'video_path': path, 'pe_0': ..., 'pe_1': ..., ...}
        """

        if not os.path.exists(video_path):
            return {"video_path": video_path,
                    **{f"pe_{i}": 0.0 for i in range(self.n_components)}}

        self._load_model()

        # Read frames
        try:
            frames = self._read_frames(video_path)
        except Exception as e:
            # Broken video or decoding error
            print(f"Error reading frames for {video_path}: {e}")
            return {"video_path": video_path,
                    **{f"pe_{i}": 0.0 for i in range(self.n_components)}}

        # Preprocess frames and move to GPU
        imgs = [self.preprocess(Image.fromarray(f)) for f in frames]
        video_tensor = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)

        # Encode video using PE
        with torch.no_grad():
            feats = self.model.encode_video(video_tensor)
            feats = feats.squeeze(0)

        feats = feats.cpu().numpy()

        # Resize/pad to desired dimension
        if len(feats) < self.n_components:
            feats = np.pad(feats, (0, self.n_components - len(feats)))
        else:
            feats = feats[:self.n_components]

        return {
            "video_path": video_path,
            **{f"pe_{i}": float(feats[i]) for i in range(self.n_components)}
        }
