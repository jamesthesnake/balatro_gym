import trlx, torch
from trlx.data.configs import TRLConfig
import pufferlib.emulation as puf
from balatro_gym.env import EightCardDrawEnv
from math import sqrt

# 16 parallel game instances
env = puf.Gymnasium(EightCardDrawEnv, num_envs=16)

def reward_fn(prompts, outputs):
    scores = []
    for prompt, out in zip(prompts, outputs):
        res = env.simulate(prompt, out)      # env wrapper must expose .simulate
        raw = res["balatro_raw_score"]       # integer
        scores.append(sqrt(raw) / 100.0 - 0.01)   # shaped reward
    return scores

cfg = TRLConfig.load_yaml("cfg/trlx_ppo.yml")

trlx.train(
    reward_fn=reward_fn,
    prompts=["<GAME>"] * cfg.train.total_steps,   # dummy; env provides obs
    config=cfg
)
