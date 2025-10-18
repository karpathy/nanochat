"""
Profiling utilities for nanochat training.

Provides unified profiling interface that activates both PyTorch profiler (traces) 
and CUDA memory snapshots for different training phases.
"""

import os
from datetime import datetime
import torch
from torch.profiler import profile, ProfilerActivity
from typing import Optional, Callable


class ProfilingManager:
    """Manages profiling for different phases of training."""
    
    # ANSI color codes
    CYAN = "\033[36m"
    RESET = "\033[0m"
    
    def __init__(
        self,
        base_dir: str,
        ddp_local_rank: int,
        master_process: bool,
        enable_profiling: bool = False,
        print_fn: Callable = print,
    ):
        self.base_dir = base_dir
        self.ddp_local_rank = ddp_local_rank
        self.master_process = master_process
        self.enable_profiling = enable_profiling
        self.max_mem_events_per_snapshot = 200000
        self.print_fn = print_fn
        
        # Colored prefix for profiler messages
        self.prefix = f"{self.CYAN}[PROFILER]{self.RESET}"
        
        # Create timestamped subdirectory for this run to avoid overwriting previous profiles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.profile_dir = os.path.join(base_dir, "profile_traces", timestamp)
        if enable_profiling and master_process:
            os.makedirs(self.profile_dir, exist_ok=True)
            self.print_fn(f"{self.prefix} Output directory: {self.profile_dir}")
        
        self.active_profiler: Optional[profile] = None
        self.cuda_memory_recording = False
        self.current_phase: Optional[str] = None  # Track current profiling phase
        
        # Stage mapping for ordered file naming
        self.stage_map = {
            "model_init": "stage1",
            "eval_bpb": "stage2",
            "training_microsteps": "stage3",
            "optimizer_step": "stage4",
        }
    
    def _get_stage_prefix(self, phase_name: str) -> str:
        """Get stage prefix for a phase name."""
        return self.stage_map.get(phase_name, "stageX")
    
    def _get_log_prefix(self, phase_name: str) -> str:
        """Get colored log prefix with phase name."""
        return f"{self.CYAN}[PROFILER:{phase_name}]{self.RESET}"
        
    def start_torch_profiler(self, phase_name: str, warmup: int = 0, active: int = 1, repeat: int = 1):
        """
        Start PyTorch profiler for a specific phase.
        
        Args:
            phase_name: Name of the profiling phase (e.g., "model_init", "eval_bpb")
            warmup: Number of warmup steps before active profiling (default: 0)
            active: Number of active profiling steps that capture traces (default: 1)
            repeat: Number of times to repeat the profiling cycle (default: 1)
        
        The profiler runs according to its schedule (warmup + active steps) and
        auto-completes after the scheduled steps. No need to call stop() explicitly.
        Call step_torch_profiler() on each iteration to advance the profiler.
        """
        if not self.enable_profiling or not self.master_process:
            return None
        
        stage_prefix = self._get_stage_prefix(phase_name)
        log_prefix = self._get_log_prefix(phase_name)
            
        def trace_handler(p):
            output_path = os.path.join(self.profile_dir, f"{stage_prefix}-{phase_name}_trace.json")
            self.print_fn(f"{log_prefix} Exporting Chrome trace to: {output_path}")
            p.export_chrome_trace(output_path)
            memory_path = os.path.join(self.profile_dir, f"{stage_prefix}-{phase_name}_memory_timeline.html")
            self.print_fn(f"{log_prefix} Exporting memory timeline to: {memory_path}")
            p.export_memory_timeline(memory_path, device=f"cuda:{self.ddp_local_rank}")
            self.print_fn(f"{log_prefix} Trace export complete")
        
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=repeat),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        )
        prof.start()
        self.print_fn(f"{log_prefix} Torch profiler started: warmup={warmup}, active={active}, repeat={repeat}")
        self.active_profiler = prof
        self.current_phase = phase_name  # Track current phase
        return prof
    
    def stop_torch_profiler(self):
        """
        Stop the active PyTorch profiler (optional - profiler stops automatically after active steps).
        Only needed if you want to stop profiling early before the schedule completes.
        """
        if self.active_profiler is not None:
            log_prefix = self._get_log_prefix(self.current_phase) if self.current_phase else self.prefix
            self.active_profiler.stop()
            self.print_fn(f"{log_prefix} Torch profiler stopped (early)")
            self.active_profiler = None
            self.current_phase = None  # Clear current phase
    
    def step_torch_profiler(self):
        """Step the active PyTorch profiler."""
        if self.active_profiler is not None:
            self.active_profiler.step()
    
    def start_cuda_memory_recording(self, phase_name: Optional[str] = None):
        """Start CUDA memory snapshot recording."""
        if not self.enable_profiling or not self.master_process:
            return
        
        if not self.cuda_memory_recording:
            if phase_name:
                self.current_phase = phase_name  # Track phase if provided
            log_prefix = self._get_log_prefix(self.current_phase) if self.current_phase else self.prefix
            self.print_fn(f"{log_prefix} Starting CUDA memory snapshot recording with max_entries={self.max_mem_events_per_snapshot}")
            torch.cuda.memory._record_memory_history(
                max_entries=self.max_mem_events_per_snapshot
            )
            self.cuda_memory_recording = True
    
    def dump_cuda_memory_snapshot(self, phase_name: str):
        """Dump and stop CUDA memory snapshot."""
        if not self.enable_profiling or not self.master_process:
            return
        
        if self.cuda_memory_recording:
            stage_prefix = self._get_stage_prefix(phase_name)
            log_prefix = self._get_log_prefix(phase_name)
            snapshot_path = os.path.join(self.profile_dir, f"{stage_prefix}-{phase_name}_mem.pickle")
            self.print_fn(f"{log_prefix} Dumping CUDA memory snapshot to: {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            self.print_fn(f"{log_prefix} Stopping CUDA memory snapshot recording")
            torch.cuda.memory._record_memory_history(enabled=None)
            self.cuda_memory_recording = False
            self.print_fn(f"{log_prefix} CUDA memory snapshot complete")
    
    def profile_section(self, phase_name: str, warmup: int = 0, active: int = 1, repeat: int = 1):
        """
        Context manager for profiling a section of code.
        
        Args:
            phase_name: Name of the profiling phase
            warmup: Number of warmup steps before active profiling (default: 0)
            active: Number of active profiling steps that capture traces (default: 1)
            repeat: Number of times to repeat the profiling cycle (default: 1)
        
        Usage:
            with profiler.profile_section("model_loading", warmup=0, active=1):
                # code to profile
                pass
        """
        return ProfilingContext(self, phase_name, warmup, active, repeat)


class ProfilingContext:
    """Context manager for profiling a specific section."""
    
    def __init__(self, manager: ProfilingManager, phase_name: str, warmup: int, active: int, repeat: int):
        self.manager = manager
        self.phase_name = phase_name
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
    
    def __enter__(self):
        self.manager.start_cuda_memory_recording(self.phase_name)
        self.manager.start_torch_profiler(self.phase_name, warmup=self.warmup, active=self.active, repeat=self.repeat)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Profiler auto-completes after scheduled steps, no need to stop explicitly
        self.manager.dump_cuda_memory_snapshot(self.phase_name)
    
    def step(self):
        """Call this at the end of each iteration in the profiled section."""
        self.manager.step_torch_profiler()
        # Profiler will auto-complete after warmup + active steps

