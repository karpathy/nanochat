"""Serialisable state for the SFT data generator."""
from dataclasses import dataclass

from nanochat.common import ddp_rank_position

# A batch with no supervised tokens gives loss 0.0 and zero gradients. Halt after this many
# optimizer steps in a row with none, so a stalled data loader cannot burn compute silently.
ZERO_TOKEN_STREAK_LIMIT = 10


def is_packable(ids: list[int], row_capacity: int) -> bool:
    """True when a conversation fits in one packed row. Longer ones can never be placed."""
    return len(ids) <= row_capacity


def next_zero_token_streak(streak: int, supervised_tokens: int) -> int:
    """Count consecutive optimizer steps with no supervised tokens; any token resets the count."""
    return streak + 1 if supervised_tokens == 0 else 0


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
    conv_buffer: list[list[list[int]]]  # list of [token_ids, loss_mask] pairs (JSON-serialisable)

    def to_dict(self):
        return {
            "cursor": self.cursor, "consumed": self.consumed,
            "epoch": self.epoch, "it": self.it, "conv_buffer": self.conv_buffer,
        }

    @classmethod
    def initial(cls, ddp_rank):
        return cls(cursor=ddp_rank, consumed=ddp_rank, epoch=1, it=0, conv_buffer=[])

    @classmethod
    def from_dict(cls, d, ddp_rank: int, ddp_world_size: int):
        # Re-derive each rank's cursor/consumed from rank-0's saved value via
        # ddp_rank_position (the single SSOT for DDP stride-block arithmetic).
        # Only applies when the key was actually saved; defaults are already rank-specific.
        if "cursor" in d:
            cursor = ddp_rank_position(d["cursor"], ddp_world_size, ddp_rank)
        else:
            cursor = ddp_rank
        if "consumed" in d:
            consumed = ddp_rank_position(d["consumed"], ddp_world_size, ddp_rank)
        else:
            consumed = ddp_rank
        return cls(
            cursor=cursor,
            consumed=consumed,
            epoch=d.get("epoch", 1),
            it=d.get("it", 0),
            conv_buffer=list(d.get("conv_buffer", [])) if ddp_rank == 0 else [],
        )
