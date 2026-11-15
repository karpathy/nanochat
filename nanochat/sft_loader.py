"""Serialisable state for the SFT data generator."""
from dataclasses import dataclass


@dataclass
class SftLoaderState:
    """Snapshot of the SFT generator's position (taken before each batch).

    Storing it in the checkpoint meta allows a resumed run to replay the same
    batch that was prefetched before the interruption.
    """
    cursor: int
    consumed: int
    epoch: int
    it: int
    conv_buffer: list  # list of [token_ids, loss_mask] pairs (JSON-serialisable)

    def to_dict(self):
        return {
            "cursor": self.cursor, "consumed": self.consumed,
            "epoch": self.epoch, "it": self.it, "conv_buffer": self.conv_buffer,
        }

    @classmethod
    def initial(cls, ddp_rank):
        return cls(cursor=ddp_rank, consumed=ddp_rank, epoch=1, it=0, conv_buffer=[])

    @classmethod
    def from_dict(cls, d, ddp_rank, ddp_world_size: int = 1):
        # Re-derive each rank's cursor/consumed from rank-0's saved value, mirroring
        # dataloader._start_position so DDP ranks don't all resume on rank-0's stream.
        # Only applies when the key was actually saved; defaults are already rank-specific.
        ws = ddp_world_size
        if "cursor" in d:
            cursor = (d["cursor"] // ws) * ws + ddp_rank
        else:
            cursor = ddp_rank
        if "consumed" in d:
            consumed = (d["consumed"] // ws) * ws + ddp_rank
        else:
            consumed = ddp_rank
        return cls(
            cursor=cursor,
            consumed=consumed,
            epoch=d.get("epoch", 1),
            it=d.get("it", 0),
            conv_buffer=list(d.get("conv_buffer", [])) if ddp_rank == 0 else [],
        )
