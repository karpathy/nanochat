

from nanochat.config import Config
from nanochat.common import report_dir, get_dist_info
from nanochat.report.base import Report

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


class DummyReport:
    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def reset(self, *args: object, **kwargs: object) -> None:
        pass

def get_report(base_dir:str):
    # just for convenience, only rank 0 logs to report
    _, ddp_rank, _, _ = get_dist_info()
    if ddp_rank == 0:
        return Report(report_dir(base_dir))
    else:
        return DummyReport()

def manage_report(config: Config, command: str = "generate"):
    match command:
        case "generate":
            get_report(config.common.base_dir).generate()
        case "reset":
            get_report(config.common.base_dir).reset()
        case _:
            raise ValueError(f"Unknown command: {command}")
    
