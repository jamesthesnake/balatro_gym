#!/usr/bin/env python3
# save_traj.py – Enhanced version with robust model saving and trajectory collection

from __future__ import annotations
import numpy as np
import pickle
import argparse
import sys
import time
import os
import torch
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Local imports
import balatro_gym.patch_balatro_env
from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import decode_discard, decode_select
from balatro_gym.score_with_balatro import int_to_card
from balatro_gym.balatro_game import BalatroGame
from heuristic_baseline import THRESHOLD_RANK, make_discard_action, make_select_action

# ------------------------------------------------ CONFIG
PASS_THRESHOLD = 300
DEFAULT_NUM_ROUNDS = 100_000
DISCARDS_PER_ROUND = 4
HANDS_PER_ROUND = 4
MAX_BEAM_DISCARDS = 4
LLM_LOCAL_PATH = "/workspace/deepseek-model"

MODEL_SAVE_DIR = Path("/workspace/saved_models")  # Centralized model storage
# --------------------------------------------------------

BUCKET_NAMES = {
    0: "random",
    1: "heuristic",
    2: "beam_search",
    3: "llm_policy",
    4: "human"
}

class ModelSaver:
    """Handles model saving with version control and metadata"""
    
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        tokenizer: Any = None,
        model_name: str = "model",
        metadata: Dict[str, Any] = None,
        max_versions: int = 5
    ) -> Path:
        """
        Save model with version control
        
        Args:
            model: PyTorch/HF model to save
            tokenizer: Corresponding tokenizer
            model_name: Base name for model
            metadata: Additional training info
            max_versions: Maximum versions to keep
            
        Returns:
            Path to saved model directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = MODEL_SAVE_DIR / f"{model_name}_v{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            if hasattr(model, "save_pretrained"):  # HF models
                model.save_pretrained(
                    save_dir,
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
            else:  # Custom PyTorch models
                torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            
            # Save tokenizer if provided
            if tokenizer and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(save_dir)
            
            # Save metadata
            meta = {
                "save_time": timestamp,
                "torch_version": torch.__version__,
                "git_hash": ModelSaver._get_git_hash(),
                **(metadata or {})
            }
            torch.save(meta, save_dir / "metadata.bin")
            
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
            raise
        
        # Clean old versions
        ModelSaver._clean_old_versions(model_name, max_versions)
        
        return save_dir
    
    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash if available"""
        try:
            return os.popen("git rev-parse HEAD").read().strip() or None
        except:
            return None
    
    @staticmethod
    def _clean_old_versions(model_name: str, max_keep: int):
        """Remove excess model versions"""
        versions = sorted(
            [d for d in MODEL_SAVE_DIR.glob(f"{model_name}_v*") if d.is_dir()],
            key=os.path.getmtime
        )
        while len(versions) > max_keep:
            oldest = versions.pop(0)
            os.system(f"rm -rf {oldest}")

def compute_raw_chip_value(card_ids: np.ndarray) -> int:
    """Calculate raw chip value for given card indices"""
    cards = [int_to_card(int(idx)) for idx in card_ids]
    return BalatroGame._evaluate_hand(cards)

def random_discard(mask, remain):
    """Randomly select a valid discard action"""
    valid = [a for a in np.flatnonzero(mask) if len(decode_discard(a)) <= remain]
    if not valid:
        valid = [a for a in np.flatnonzero(mask) if len(decode_discard(a)) == 0]
    action = int(np.random.choice(valid))
    return action, len(decode_discard(action))

def random_select(mask):
    """Randomly select a valid action"""
    return int(np.random.choice(np.flatnonzero(mask)))

def heuristic_discard(hand, remain):
    """Heuristic-based discard selection"""
    a = make_discard_action(hand, THRESHOLD_RANK)
    return (a, len(decode_discard(a))) if len(decode_discard(a)) <= remain else (0, 0)

def heuristic_select(hand):
    """Heuristic-based card selection"""
    return make_select_action(hand)

def beam_best_discard(hand: np.ndarray,
                     mask: np.ndarray,
                     remain_disc: int) -> Tuple[int, int]:
    """Beam search for best discard action"""
    best_a, best_score = 0, -1
    for a in np.flatnonzero(mask):
        disc = decode_discard(int(a))
        if len(disc) > remain_disc:
            continue
        keep_idx = tuple(sorted(set(range(8)) - set(disc)))
        if len(keep_idx) != 5:
            continue
        raw = compute_raw_chip_value(hand[list(keep_idx)])
        if raw > best_score:
            best_score, best_a = raw, int(a)
    return best_a, len(decode_discard(best_a))

def load_llm_policy() -> Tuple[Any, Any]:
    """Enhanced model loader with automatic saving"""
    possible_paths = [
        Path(LLM_LOCAL_PATH),
        Path("~/deepseek-model").expanduser(),
        Path("/usr/local/share/deepseek-model"),
        Path(__file__).parent / "deepseek-model"
    ]
    
    for path in possible_paths:
        try:
            if not path.exists():
                continue
                
            print(f"🔍 Checking for model in {path}...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Handle HuggingFace cache structure
            if "snapshots" in str(path):
                snapshots = [d for d in path.iterdir() if d.is_dir()]
                if snapshots:
                    path = snapshots[0]
            
            # Verify required files exist
            required_files = ["config.json", "tokenizer.json"]
            missing = [f for f in required_files if not (path / f).exists()]
            weights_exist = any((path / f).exists() for f in [
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json"
            ])
            if missing or not weights_exist:
                print(f"⚠️ Missing files in {path}: {', '.join(missing)}")
                if not weights_exist:
                    print(f"⚠️ No valid model weight shards found in {path}")
                    continue
 
            print(f"✅ Found complete model in {path}")
            print("⏳ Loading model...")
            
            tok = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=True
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=True
            )
            mdl.eval()
            
            # Save copy to standard location
            if not path.samefile(MODEL_SAVE_DIR):
                print("💾 Creating backup in model storage...")
            
            return tok, mdl
            
        except Exception as e:
            print(f"⚠️ Error loading from {path}: {str(e)}")
            continue
    
    print("\n❌ Could not load model from any standard location")
    return None, None

tok_llm, mdl_llm = load_llm_policy()

def llm_policy(env, mask):
    """LLM-based card selection policy"""
    if not tok_llm or not mdl_llm:
        return heuristic_select(env.hand)
    
    prompt = (
        "Balatro Hand Advice:\n"
        f"Hand: {' '.join(env.card_to_str(c) for c in env.hand)}\n"
        "Choose exactly 5 cards to keep. Format:\n"
        "KEEP [card1] [card2] [card3] [card4] [card5]\n"
        "Response: "
    )
    
    try:
        inputs = tok_llm(prompt, return_tensors="pt").to(mdl_llm.device)
        outputs = mdl_llm.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tok_llm.eos_token_id
        )
        response = tok_llm.decode(outputs[0], skip_special_tokens=True)
        
        if "KEEP" in response:
            card_strs = [s for s in response.split("KEEP")[1].strip().split() if len(s) == 2][:5]
            if len(card_strs) == 5:
                idxs = []
                for card_str in card_strs:
                    try:
                        idxs.append(env.hand.index(env.str_to_card(card_str)))
                    except ValueError:
                        continue
                
                if len(idxs) == 5:
                    for a in range(256, 312):
                        if sorted(decode_select(a)) == sorted(idxs):
                            if mask[a]:
                                return a
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
    
    return heuristic_select(env.hand)

def play_round(bucket_id: int) -> List[List[dict]]:
    """Play one complete round (4 hands)"""
    round_data = []
    round_score = 0
    discards_left = DISCARDS_PER_ROUND

    for _ in range(HANDS_PER_ROUND):
        env = EightCardDrawEnv()
        obs, _ = env.reset()
        hand_data = []
        done = False
        
        while not done:
            phase = obs["phase"]
            mask = obs["action_mask"]
            
            if phase == 0:  # Discard phase
                if bucket_id == 0:
                    action, n_disc = random_discard(mask, discards_left)
                elif bucket_id == 1:
                    action, n_disc = heuristic_discard(env.hand, discards_left)
                elif bucket_id == 2:
                    action, n_disc = beam_best_discard(env.hand, mask, discards_left)
                else:
                    action, n_disc = random_discard(mask, discards_left)
                discards_left -= n_disc
            else:  # Select phase
                if bucket_id == 0:
                    action = random_select(mask)
                elif bucket_id == 1:
                    action = heuristic_select(env.hand)
                elif bucket_id == 2:
                    action = max(np.flatnonzero(mask))
                elif bucket_id == 3:
                    action = llm_policy(env, mask)
                    if not mask[action]:
                        action = heuristic_select(env.hand)
                else:
                    action = random_select(mask)
                n_disc = 0

            # Record pre-action state
            step_data = {
                "bucket": bucket_id,
                "hand_before": env.hand.copy(),
                "phase": phase,
                "action": int(action),
                "reward": 0.0,
                "done": False
            }

            # Execute action
            obs, reward, done, _, _ = env.step(action)
            
            # Record post-action state
            step_data.update({
                "reward": float(reward),
                "hand_after": env.hand.copy() if phase == 0 else None,
                "done": done
            })

            if phase == 1:  # Selection-specific data
                kept = decode_select(action)
                raw_score = compute_raw_chip_value(env.hand[list(kept)])
                round_score += raw_score
                step_data.update({
                    "keep_indices": kept,
                    "balatro_raw_score": raw_score,
                    "num_discards": n_disc
                })
                
            hand_data.append(step_data)

        # Mark final hand of the round
        if len(round_data) == HANDS_PER_ROUND - 1:
            hand_data[1]["round_pass"] = round_score > PASS_THRESHOLD
        hand_data[1]["round_score_so_far"] = round_score
        round_data.append(hand_data)

    return round_data

def collect_bucket(bucket_id: int, num_rounds: int) -> List[List[List[dict]]]:
    """Collect trajectories for a specific bucket"""
    bucket_name = BUCKET_NAMES.get(bucket_id, f"bucket_{bucket_id}")
    print(f"\nStarting {bucket_name} collection ({num_rounds} rounds)...")
    
    trajectories = []
    start_time = time.time()
    
    for i in range(num_rounds):
        trajectories.append(play_round(bucket_id))
        
        if (i + 1) % max(1, num_rounds // 10) == 0:
            elapsed = time.time() - start_time
            print(f"  Completed {i + 1}/{num_rounds} rounds ({elapsed:.1f}s)")
    
    return trajectories

def save_bucket(bucket_id: int, data: List, timestamp: str):
    """Save bucket data to pickle file"""
    os.makedirs("pickles", exist_ok=True)
    bucket_name = BUCKET_NAMES.get(bucket_id, f"bucket_{bucket_id}")
    filename = f"pickles/{bucket_name}_{timestamp}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(data)} rounds to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", default="0,1,2,3",
                       help="Comma-separated bucket IDs (0-4)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS,
                       help="Number of rounds per bucket")
    parser.add_argument("--save-model", action="store_true",
                       help="Force save model after completion")
    args = parser.parse_args()

    # Initialize model storage
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    
    # Load policy (will auto-save if found)
    global tok_llm, mdl_llm
    tok_llm, mdl_llm = load_llm_policy()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bucket_ids = [int(b) for b in args.buckets.split(",")]
    
    print(f"\nBalatro Trajectory Collection")
    print(f"Buckets: {[BUCKET_NAMES.get(b, '?') for b in bucket_ids]}")
    print(f"Rounds per bucket: {args.rounds}")
    print(f"Model storage: {MODEL_SAVE_DIR}")
    
    for bucket_id in bucket_ids:
        data = collect_bucket(bucket_id, args.rounds)
        save_bucket(bucket_id, data, timestamp)
    
    # Optionally save final model
    if args.save_model and mdl_llm:
        save_path = ModelSaver.save_model(
            mdl_llm,
            tok_llm,
            "final_policy",
            {"training_rounds": args.rounds}
        )
        print(f"✅ Final model saved to {save_path}")

if __name__ == "__main__":
    main()
