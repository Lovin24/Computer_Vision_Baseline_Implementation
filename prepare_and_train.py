import os
import json
import csv
from pathlib import Path
import subprocess

def main():
    captions_dir = Path("data/Captions")
    out_json = Path("data/all_captions.json")
    
    print("1. Parsing CSV files from data/Captions...")
    all_data = []
    
    for csv_file in captions_dir.glob("*.csv"):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # First column is video_id, e.g. Abuse018_x264.mp4
                video_file = row[0]
                video_id = video_file.replace('.mp4', '')
                
                # Remaining columns are captions
                for cap in row[1:]:
                    cap = cap.strip()
                    if cap:
                        all_data.append({
                            "video_id": video_id,
                            "caption": cap
                        })
                        
    print(f"Found {len(all_data)} caption pairs.")
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4)
        
    print("Saved to data/all_captions.json")
    
    # We remove the dummy vocab so train_captioning.py rebuilds it with actual data
    if os.path.exists("data/vocab.json"):
        os.remove("data/vocab.json")
        print("Removed dummy vocab.json. A new one will be built.")
        
    print("\n2. Starting Training for 5 epochs to demonstrate...")
    # Run the training script
    cmd = [
        "python", "src/train_captioning.py",
        "--captions_file", str(out_json),
        "--features_dir", "data/features/i3d",
        "--epochs", "5",
        "--batch_size", "32",
        "--save_path", "data/weights/caption_transformer.pth"
    ]
    
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
