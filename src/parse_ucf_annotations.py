import json
import logging
from pathlib import Path
import cv2
import math
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger(__name__)

NUM_SEGMENTS = 32

def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def main():
    repo_root = Path(__file__).resolve().parent.parent
    annotations_file = repo_root / "data" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    train_dir = repo_root / "data" / "Train"
    testing_normals_dir = repo_root / "data" / "Testing_Normal_Videos"
    
    if not annotations_file.is_file():
        log.error(f"Cannot find {annotations_file}")
        return

    # Check for both Testing_Normal_Videos and its nested possibility Testing_Normal_Videos/Testing_Normal_Videos
    if (testing_normals_dir / "Testing_Normal_Videos_Anomaly").is_dir():
        testing_normals_dir = testing_normals_dir / "Testing_Normal_Videos_Anomaly"

    output_json = repo_root / "data" / "test_annotations.json"
    
    out_dict = {}
    missing_videos = []
    
    with open(annotations_file, "r") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Processing videos"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
            
        video_file = parts[0]
        category = parts[1]
        start1, end1 = int(parts[2]), int(parts[3])
        start2, end2 = int(parts[4]), int(parts[5])
        
        # Determine paths
        video_path = None
        if category == "Normal":
            # Search in testing normals
            c1 = testing_normals_dir / video_file
            if c1.is_file():
                video_path = c1
            else:
                c2 = train_dir / "NormalVideos" / video_file
                if c2.is_file():
                    video_path = c2
        else:
            c3 = train_dir / category / video_file
            if c3.is_file():
                video_path = c3
                
        if video_path is None:
            missing_videos.append(video_file)
            continue
            
        # Feature filename expected by evaluate_mil: e.g. Abuse028_x264_C
        base_name = Path(video_file).stem
        if category == "Normal":
            base_name = base_name.replace("Normal_Videos_", "Normal_Videos")
        target_name = f"{base_name}_C"
        
        total_frames = get_video_frame_count(video_path)
        if total_frames <= 0:
            log.warning(f"Could not read frames for {video_path}")
            continue
            
        # Map anomalous frames to 32 segments
        # L is segment length
        # Segment integer = floor((f / total_frames) * 32)
        anomalous_segments = set()
        
        def add_range(start_f, end_f):
            if start_f == -1 or end_f == -1:
                return
            start_seg = math.floor((start_f / total_frames) * NUM_SEGMENTS)
            end_seg = math.floor((end_f / total_frames) * NUM_SEGMENTS)
            # Clip limits to valid indices
            start_seg = max(0, min(start_seg, NUM_SEGMENTS - 1))
            end_seg = max(0, min(end_seg, NUM_SEGMENTS - 1))
            for s in range(start_seg, end_seg + 1):
                anomalous_segments.add(s)
                
        add_range(start1, end1)
        add_range(start2, end2)
        
        out_dict[target_name] = sorted(list(anomalous_segments))

    with open(output_json, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    log.info(f"Saved {len(out_dict)} annotations to {output_json}")
    if missing_videos:
        log.warning(f"Failed to find {len(missing_videos)} videos: {missing_videos[:10]}")

if __name__ == "__main__":
    main()
