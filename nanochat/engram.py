"""
Engram: hashed n-gram lookup tables in host memory, injected into attention values.
Notable features:
- one bf16 table in /dev/shm, shared by all ranks and updated lock-free (Hogwild!)
- each position hashes its trailing n-grams (several salted slots) into rows of the table
- every engram layer reads its slice of the gathered rows, gates it on a few residual
  channels, projects it up to the model dim and adds it to that layer's attention values
- the gathered rows are a GPU autograd leaf; step() applies a row-wise RMSprop update on
  GPU and writes the rows back to the host asynchronously (stochastic rounding to bf16)
- the next micro-step's rows are gathered on a worker thread during the current backward
Hashes roll across document boundaries within a packed sequence (known approximation).
"""

import concurrent.futures
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRIMES = (1_000_003, 998_244_353, 911_382_323, 805_306_457)


def _fused_update_impl(grad, vals, accum_rows, lr, n_layer, n_embd, eps, decay, accum_beta):
    """Row-wise RMSprop + stochastic rounding to bf16, one kernel."""
    g = grad.view(-1, n_layer, n_embd)
    accum_rows = accum_beta * accum_rows + (1.0 - accum_beta) * g.pow(2).mean(dim=-1)
    scale = lr / (accum_rows.sqrt() + eps)
    x = vals.float() * (1.0 - decay) - (scale.unsqueeze(-1) * g).view(-1, n_layer * n_embd)
    # stochastic rounding: round-to-nearest would drop every update below the bf16 ulp
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    out = ((x.view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    return out, accum_rows


_FUSED_UPDATE = None


def _fused_update(*args):
    global _FUSED_UPDATE
    if _FUSED_UPDATE is None:
        try:
            _FUSED_UPDATE = torch.compile(_fused_update_impl, dynamic=True) # row count changes every step
        except Exception:
            _FUSED_UPDATE = _fused_update_impl
    try:
        return _FUSED_UPDATE(*args)
    except Exception:
        _FUSED_UPDATE = _fused_update_impl
        return _fused_update_impl(*args)


def largest_prime_at_most(n):
    """Prime table sizes spread multiplicative hashes much better than powers of two."""
    def is_prime(m):
        if m < 2 or m % 2 == 0:
            return m == 2
        f = 3
        while f * f <= m:
            if m % f == 0:
                return False
            f += 2
        return True
    n = int(n) | 1
    while n > 2 and not is_prime(n):
        n -= 2
    return n


def ngram_hash(idx, table_size, ngram, salt=0):
    """Hash each position's trailing `ngram` token ids (B, T) into bucket ids in [0, table_size)."""
    assert 1 <= ngram <= len(_PRIMES)
    acc = idx * (_PRIMES[0] + salt * 2_654_435_761)
    for j in range(1, ngram):
        prev = torch.zeros_like(idx) # missing history (row start, single-token decode) hashes as token 0
        if j < idx.size(1):
            prev[:, j:] = idx[:, :-j]
        acc = acc + prev * (_PRIMES[j] + salt * 40_503)
    return acc.remainder(table_size)


class EngramBank(nn.Module):
    """The host-resident table (rows x n_layer*n_embd) and its row-wise optimizer."""

    def __init__(self, table_size, n_layer, n_embd, decay, accum_beta):
        super().__init__()
        self.table_size = table_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.decay = decay
        self.accum_beta = accum_beta
        # not persistent: checkpoints hold the model only, the table stays in host memory
        self.register_buffer("weight", torch.zeros(table_size, n_layer * n_embd, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("adagrad_accum", torch.zeros(table_size, n_layer), persistent=False)
        self._pending = [] # (uniq_idx, leaf) per micro-step, consumed by step()
        self._stage_in = None # pinned staging buffers, sized lazily
        self._stage_out = None
        self._pf_slots = [None, None] # double-buffered prefetch
        self._pf_events = [None, None]
        self._pf_next_slot = 0
        self._pf_pending = None
        self._exec = None # single worker thread => FIFO order over all host-table accesses
        self._side = None # side CUDA stream for H2D / D2H
        self._write_future = None
        self._d2h_event = None
        self.shared_path = None

    def _worker(self):
        if self._exec is None:
            self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-host")
        return self._exec

    def _side_stream(self, device):
        if self._side is None:
            self._side = torch.cuda.Stream(device=device)
        return self._side

    def flush(self):
        """Block until the async host write has landed."""
        f, self._write_future = self._write_future, None
        if f is not None:
            f.result()

    def shutdown(self, unlink_shared=False):
        self.flush()
        if self._exec is not None:
            self._exec.shutdown(wait=True)
            self._exec = None
        if unlink_shared and self.shared_path:
            self.weight = torch.empty(0) # tmpfs pages are freed once the file and every mapping are gone
            try:
                os.unlink(self.shared_path)
            except FileNotFoundError:
                pass

    def init_weights(self, shared_path, rank=0):
        """Map the table from one tmpfs file shared by every rank. Zero table => no-op at step 0."""
        self.flush()
        need = self.table_size * self.n_layer * self.n_embd * 2
        st = os.statvfs(os.path.dirname(shared_path) or "/")
        avail = st.f_bavail * st.f_frsize
        assert need < 0.9 * avail, f"engram table needs {need/2**30:.1f} GB but {shared_path} has {avail/2**30:.1f} GB free"
        self.weight = torch.empty(0) # drop the to_empty() storage before mapping the host table
        fresh = not os.path.exists(shared_path) or os.path.getsize(shared_path) == 0
        n = self.table_size * self.n_layer * self.n_embd
        self.weight = torch.from_file(shared_path, shared=True, size=n, dtype=torch.bfloat16).view(self.table_size, -1)
        if rank == 0 and not fresh: # a freshly created file is sparse and already reads as zero
            self.weight.zero_()
        self.shared_path = shared_path
        self.adagrad_accum.zero_()
        self._stage_in = None
        self._stage_out = None
        self._pf_slots = [None, None]
        self._pf_events = [None, None]
        self._pf_pending = None

    def _apply(self, fn, *args, **kwargs):
        # keep the table out of .to()/.cuda()/to_empty(): it would otherwise be materialized in VRAM
        saved = self.weight
        self.weight = torch.empty(0)
        super()._apply(fn, *args, **kwargs)
        self.weight = saved
        return self

    def _staging(self, name, rows):
        buf = getattr(self, name, None)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty(rows, self.n_layer * self.n_embd, dtype=torch.bfloat16).pin_memory()
            setattr(self, name, buf)
        return buf[:rows]

    @torch._dynamo.disable
    def prefetch(self, buckets):
        """Gather the rows for a future forward on the worker thread, overlapping the current backward."""
        if self.weight.numel() == 0 or buckets is None:
            return
        device = buckets.device
        uniq, inv = torch.unique(buckets.reshape(-1), return_inverse=True)
        uniq_cpu = uniq.to("cpu")
        n = uniq_cpu.numel()
        slot = self._pf_next_slot
        self._pf_next_slot ^= 1
        buf = self._pf_slots[slot]
        if buf is None or buf.shape[0] < n:
            # pin_memory() device-synchronizes, so allocate on the main thread, never in the worker
            buf = torch.empty(n, self.n_layer * self.n_embd, dtype=torch.bfloat16).pin_memory()
            self._pf_slots[slot] = buf
            self._pf_events[slot] = None
        ev = self._pf_events[slot] or torch.cuda.Event()
        self._pf_events[slot] = ev
        self._pf_pending = dict(buckets=buckets, uniq=uniq, uniq_cpu=uniq_cpu, inv=inv,
                                stage=buf[:n], event=ev, device=device, future=None, gpu_rows=None)
        self._pf_pending["future"] = self._worker().submit(self._gather_task, self._pf_pending)

    def _gather_task(self, p):
        ev, stage, device = p["event"], p["stage"], p["device"]
        if ev.query() is False:
            ev.synchronize() # the previous H2D that read this slot must be done before we overwrite it
        torch.index_select(self.weight, 0, p["uniq_cpu"], out=stage)
        torch.cuda.set_device(device) # thread-local
        side = self._side_stream(device)
        with torch.no_grad(), torch.cuda.stream(side):
            gpu_rows = stage.to(device, non_blocking=True)
            ev.record(side)
        return gpu_rows

    def _take_prefetch(self, buckets):
        """The staged rows if they are for exactly these buckets, else None."""
        p, self._pf_pending = self._pf_pending, None
        if p is None:
            return None
        p["gpu_rows"] = p["future"].result()
        if p["buckets"].shape != buckets.shape or not torch.equal(p["buckets"], buckets):
            return None
        return p

    def _patch_prefetch(self, merged, new_rows):
        """A prefetch gathered before this step's write differs from the table only in the rows just
        updated, which are still on the GPU: patch them in rather than re-reading the host."""
        p = self._pf_pending
        if p is None or merged.numel() == 0:
            return
        if p["gpu_rows"] is None:
            p["gpu_rows"] = p["future"].result()
        gpu_rows = p["gpu_rows"]
        cur = torch.cuda.current_stream()
        cur.wait_event(p["event"])
        gpu_rows.record_stream(cur)
        uniq = p["uniq"]
        pos = torch.searchsorted(merged, uniq).clamp_(max=merged.numel() - 1) # both sorted (torch.unique)
        hit = merged[pos] == uniq
        if bool(hit.any()):
            gpu_rows[hit] = new_rows[pos[hit]].to(gpu_rows.dtype)

    @torch._dynamo.disable
    def _host_gather(self, uniq_gpu, device, dtype):
        self.flush() # order every synchronous read after the outstanding async write
        uniq_cpu = uniq_gpu.to("cpu")
        stage = self._staging("_stage_in", uniq_cpu.numel())
        torch.index_select(self.weight, 0, uniq_cpu, out=stage)
        return stage.to(device=device, non_blocking=True).to(dtype)

    @torch._dynamo.disable
    def lookup(self, buckets, device, dtype):
        """buckets (G, B, T) -> rows (G, B, T, n_layer*n_embd) on device, as a grad-tracked leaf."""
        training = self.training and torch.is_grad_enabled()
        p = self._take_prefetch(buckets) if training else None # eval forwards must not consume the prefetch
        if p is not None:
            uniq, inv = p["uniq"], p["inv"]
            gpu_rows = p["gpu_rows"]
            cur = torch.cuda.current_stream()
            cur.wait_event(p["event"])
            gpu_rows.record_stream(cur) # allocated on the side stream, consumed here
            leaf = gpu_rows.to(dtype).requires_grad_(True)
        else:
            uniq, inv = torch.unique(buckets.reshape(-1), return_inverse=True)
            leaf = self._host_gather(uniq, device, dtype).requires_grad_(True)
        if training:
            self._pending.append((uniq, leaf))
        return leaf[inv].view(*buckets.shape, self.n_layer * self.n_embd)

    def step(self, lr, eps=1e-8):
        """Update the rows touched since the last step (grads summed over micro-steps) and write them back."""
        if not self._pending:
            return
        device = self._pending[0][1].device
        LD = self.n_layer * self.n_embd
        pending, self._pending = self._pending, []
        merged, inv = torch.unique(torch.cat([u for u, _ in pending]), return_inverse=True)
        grad = torch.zeros(merged.numel(), LD, device=device)
        vals = torch.empty(merged.numel(), LD, dtype=pending[0][1].dtype, device=device) # the leaves hold the current row values
        off = 0
        for uniq, leaf in pending:
            sl = inv[off:off + uniq.numel()]
            if leaf.grad is not None:
                grad.index_add_(0, sl, leaf.grad.float())
            vals[sl] = leaf.detach()
            off += uniq.numel()
        accum_rows = self.adagrad_accum[merged]
        new_rows, accum_new = _fused_update(grad, vals, accum_rows, lr, self.n_layer, self.n_embd, eps, self.decay, self.accum_beta)
        self.adagrad_accum[merged] = accum_new
        # write back: D2H on the side stream, host index_copy_ on the worker (no cuda.synchronize())
        merged_cpu = merged.to("cpu")
        self.flush() # the previous write has landed, so _stage_out is free
        stage = self._staging("_stage_out", merged.numel())
        side = self._side_stream(device)
        ev = self._d2h_event or torch.cuda.Event()
        self._d2h_event = ev
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            stage.copy_(new_rows, non_blocking=True)
            ev.record(side)
        new_rows.record_stream(side)

        def _write_task(_anchor=new_rows): # the closure keeps new_rows alive until the D2H is done
            ev.synchronize()
            self.weight.index_copy_(0, merged_cpu, stage)

        self._write_future = self._worker().submit(_write_task)
        self._patch_prefetch(merged, new_rows)


class EngramLayer(nn.Module):
    """One layer's view of the bank: gate on the token's residual channels, then project the embedding up."""

    def __init__(self, layer_idx, rank, n_embd, gate_channels):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate_channels = gate_channels
        self.gate = nn.Linear(gate_channels, 1, bias=False)
        self.up = nn.Linear(rank, n_embd, bias=False)

    def init_weights(self):
        torch.nn.init.uniform_(self.gate.weight, 0.0, 0.02)
        torch.nn.init.orthogonal_(self.up.weight) # the embeddings start at zero and first learn through this basis

    def forward(self, x, bank_out):
        # x: (B, T, n_embd), bank_out: (B, T, n_layer, rank)
        e = bank_out[:, :, self.layer_idx]
        gin = x[..., :self.gate_channels]
        g = torch.sigmoid(F.linear(gin, self.gate.weight.to(gin.dtype)) / self.gate_channels) # fan-in norm: step size independent of width
        return F.linear(g * e, self.up.weight.to(e.dtype)) # gate the narrow embedding, then project
