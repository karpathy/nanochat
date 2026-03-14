"""Wandb stubs for when wandb logging is disabled."""


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures."""

    def __init__(self):
        pass

    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def finish(self):
        pass
