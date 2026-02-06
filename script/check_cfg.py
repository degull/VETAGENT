from vetagent.config import load_cfg

cfg = load_cfg("E:/VETAgent/configs/vetagent/vetagent.yaml")
print(cfg)
print("dim:", cfg.backbone.dim)
print("actions:", cfg.actions.enabled)

# cd E:\VETAgent
# python -c "from vetagent.config import load_cfg; cfg=load_cfg('E:/VETAgent/configs/vetagent/vetagent.yaml'); print(cfg.backbone.dim, cfg.actions.enabled)"
