import trlx
from trlx.data.configs import TRLConfig

cfg = TRLConfig.load_yaml("cfg/trlx_sft.yml")

trlx.train(
    config=cfg,
    dataset_path="data/balatro_sft.jsonl",
    split_percent=[95, 5]
)
