from vetagent.config import load_cfg
from vetagent.factory.build_backbone import build_backbone
from vetagent.factory.build_toolbank import build_toolbank

cfg = load_cfg("E:/VETAgent/configs/vetagent/vetagent.yaml")

model, backbone_fp, bb_meta = build_backbone(cfg, map_location="cpu")
print("Backbone meta exists:", bb_meta is not None)
print("Backbone fp:", backbone_fp)

tools, metas = build_toolbank(cfg, backbone_fp=backbone_fp, map_location="cpu")
print("Tool keys:", list(tools.keys()))
print("Tool meta exists:", {k: (v is not None) for k, v in metas.items()})
print("OK factory wiring")
