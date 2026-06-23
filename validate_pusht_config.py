#!/usr/bin/env python
"""
End-to-end validation that the pusht dataset loads correctly with RLDX-1's
LeRobotEpisodeLoader using the pusht modality config.
"""

import sys
from pathlib import Path

# Add RLDX-1 to path
rldx_root = Path(__file__).parent
sys.path.insert(0, str(rldx_root))

from rldx.configs.data.pusht_config import pusht
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

dataset_path = rldx_root / "examples" / "pusht_lerobot"

if not dataset_path.exists():
    print(f"❌ Dataset not found at {dataset_path}")
    print("   Run: uv run python download_pusht.py")
    sys.exit(1)

print(f"✓ Dataset found at {dataset_path}")

try:
    loader = LeRobotEpisodeLoader(str(dataset_path), modality_configs=pusht)
    print(f"✓ LeRobotEpisodeLoader created successfully")
    print(f"  - Number of episodes: {len(loader)}")
    print(f"  - Episode 0 length: {loader.get_episode_length(0)}")

    # Load a sample episode through the full pipeline
    episode_df = loader[0]
    print(f"\n✓ Successfully loaded episode 0")
    print(f"  - DataFrame columns: {list(episode_df.columns)}")
    print(f"  - Num frames: {len(episode_df)}")

    # Inspect shapes of each modality
    for col in episode_df.columns:
        sample = episode_df[col].iloc[0]
        if hasattr(sample, "shape"):
            print(f"  - {col}: shape={sample.shape}, dtype={getattr(sample, 'dtype', type(sample).__name__)}")
        else:
            print(f"  - {col}: {type(sample).__name__} = {sample}")

    print(f"\n✅ All validation checks passed!")
    print(f"\nReady to train. Run:")
    print(f"  bash run_scripts/train/examples/finetune_pusht.sh")

except Exception as e:
    print(f"\n❌ Validation failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
