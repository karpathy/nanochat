"""Resolve SFT defaults from a pretrained checkpoint without training dependencies."""


def inherit_sft_config(args, meta, print0):
    pretrain_user_config = meta.get("user_config", {}).copy()
    # Checkpoints store raw CLI LRs, but pretraining scales them by the resolved
    # batch size before optimizer setup. Reconstruct that scale, independently
    # of any SFT batch override. Missing historical LRs keep the defaults below.
    pretrain_batch_size = meta.get("total_batch_size")
    batch_lr_scale = (pretrain_batch_size / 524288) ** 0.5 if pretrain_batch_size is not None else 1.0
    for name in ("embedding_lr", "unembedding_lr", "matrix_lr"):
        if pretrain_user_config.get(name) is not None:
            pretrain_user_config[name] *= batch_lr_scale
    for name, fallback, source in [
        ("max_seq_len",       2048,  meta),
        ("device_batch_size", 32,    meta),
        ("total_batch_size",  524288, meta),
        ("embedding_lr",      0.3,   pretrain_user_config),
        ("unembedding_lr",    0.004, pretrain_user_config),
        ("matrix_lr",         0.02,  pretrain_user_config),
    ]:
        arg_val = getattr(args, name)
        pretrain_val = source.get(name)
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(args, name, resolved)
            print0(f"Inherited {name}={resolved} from pretrained checkpoint")
        elif pretrain_val is not None and arg_val != pretrain_val:
            print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
        else:
            print0(f"Using {name}={arg_val}")
