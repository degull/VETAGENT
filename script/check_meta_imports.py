import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from vetagent.config import load_cfg
from vetagent.meta import ckpt_meta, fingerprint, validate

cfg = load_cfg("E:/VETAgent/configs/vetagent/vetagent.yaml")
print("OK imports:", cfg.backbone.dim)
