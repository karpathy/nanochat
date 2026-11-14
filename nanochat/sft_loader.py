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
    def from_dict(cls, d, ddp_rank):
        return cls(
            cursor=d.get("cursor", ddp_rank),
            consumed=d.get("consumed", ddp_rank),
            epoch=d.get("epoch", 1),
            it=d.get("it", 0),
            conv_buffer=list(d.get("conv_buffer", [])),
        )
