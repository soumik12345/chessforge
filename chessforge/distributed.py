"""Unified JAX initialization for Colab TPU (v5e-1/v6e-1), multi-host TPU, and single/multi-GPU."""

from __future__ import annotations

import os


def configure_xla_env() -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    extra = []
    if "--xla_gpu_triton_gemm_any" not in flags:
        extra.append("--xla_gpu_triton_gemm_any=True")
    if extra:
        os.environ["XLA_FLAGS"] = (flags + " " + " ".join(extra)).strip()


def init_distributed() -> None:
    configure_xla_env()
    import jax

    if jax.distributed.is_initialized():
        return

    # 1) PyTorch / torchrun style multi-node or multi-process GPU launch
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and "MASTER_ADDR" in os.environ:
        addr = os.environ["MASTER_ADDR"]
        port = os.environ.get("MASTER_PORT", "29500")
        rank = int(os.environ["RANK"])
        local_rank = os.environ.get("LOCAL_RANK")
        local_ids = [int(local_rank)] if local_rank is not None else None
        jax.distributed.initialize(
            coordinator_address=f"{addr}:{port}",
            num_processes=world_size,
            process_id=rank,
            local_device_ids=local_ids,
        )
        return

    # 2) SLURM multi-process GPU/CPU launch
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if slurm_ntasks > 1:
        jax.distributed.initialize()
        return

    # 3) Multi-host Cloud TPU pod (Colab v5e-1 / v6e-1 is single-host and skips cleanly)
    tpu_chips_per_host = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", "")
    has_multihost_tpu = any(
        k in os.environ
        for k in ("MEGASCALE_COORDINATOR_ADDRESS", "TPU_WORKER_HOSTNAMES")
    )
    if has_multihost_tpu and "," in os.environ.get("TPU_WORKER_HOSTNAMES", ""):
        jax.distributed.initialize()
        return

    # Fallback probe for environments managed by JAX coordinator env vars
    if "JAX_COORDINATOR_ADDRESS" in os.environ or (
        tpu_chips_per_host and os.environ.get("CLOUD_TPU_TASK_ID") not in (None, "0")
    ):
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001
            print(f"[info] jax.distributed.initialize skipped: {e}")
