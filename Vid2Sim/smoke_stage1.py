# Short smoke test: Stage I only (feed-forward physics prediction + LGM GS reconstruction).
# Verifies checkpoints, data, kaolin/dgr/kiui/transformers stack on GPU without long training.
import argparse
from utils.seeding import seed_everything
# ckpt_phys_predictor.pth was pickled from a __main__ that defined these classes
from models.phys_predictor import FeedForwardPredictor, RegressionHead, LBSHead  # noqa: F401
import run_pipeline as rp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gso.yaml")
    parser.add_argument("--dataset_dir", type=str, default="dataset/GSO")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_name", type=str, default="bus")
    parser.add_argument("--ckpt_predictor", type=str, default="checkpoints/ckpt_phys_predictor.pth")
    parser.add_argument("--ckpt_lbs", type=str, default="checkpoints/ckpt_lbs_template.pth")
    parser.add_argument("--ckpt_lgm", type=str, default="checkpoints/ckpt_lgm.safetensors")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    rp.predict_phys_params(args)
    rp.predict_gs_LGM(args)
    print("SMOKE STAGE I OK")
