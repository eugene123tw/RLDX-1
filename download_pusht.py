from huggingface_hub import snapshot_download
import json
from pathlib import Path

# Download the lerobot/pusht dataset in v2.1 format (RLDX-1 expects v2.1, not v3.0)
print("Downloading lerobot/pusht dataset (v2.1 format)...")
cache_dir = snapshot_download(
    'lerobot/pusht',
    repo_type='dataset',
    revision='v2.1',  # RLDX-1's loader expects v2.1 (JSONL-based metadata)
    local_dir='./examples/pusht_lerobot',
)
print(f'Downloaded to: {cache_dir}')

# Read and display the dataset structure
info_path = Path(cache_dir) / 'meta' / 'info.json'
if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    print(f"\nDataset info:")
    print(f"  FPS: {info.get('fps')}")
    print(f"  Features: {list(info.get('features', {}).keys())}")
    print(f"\nFeature details:")
    for feature_name, feature_info in info.get('features', {}).items():
        print(f"  {feature_name}: shape={feature_info.get('shape')}, dtype={feature_info.get('dtype')}")
else:
    print("Warning: meta/info.json not found")