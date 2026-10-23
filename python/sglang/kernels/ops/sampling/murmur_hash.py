import torch
import triton
import triton.language as tl


@triton.jit
def rotl32(x, r: tl.constexpr) -> tl.uint32:
    """
    rotate left 32-bit integer x by r bits
    e.g. x = 01110001, r = 2 -> 11000101
    """
    x = x.to(tl.uint64)
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


@triton.jit
def fmix32(h: tl.uint32) -> tl.uint32:
    """
    final mix of 32-bit hash value for MurmurHash
    """
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
    """
    Mixes a 32-bit key into the hash state.
    """
    c1: tl.uint32 = 0xCC9E2D51
    c2: tl.uint32 = 0x1B873593
    r1: tl.constexpr = 15
    r2: tl.constexpr = 13
    mm: tl.uint32 = 5
    nn: tl.uint32 = 0xE6546B64

    k = (k * c1) & 0xFFFFFFFF
    k = rotl32(k, r1)
    k = (k * c2) & 0xFFFFFFFF
    h ^= k
    h = rotl32(h, r2)
    h = (h * mm + nn) & 0xFFFFFFFF
    return h


@triton.jit
def murmur_hash32_kernel(
    seed_ptr,
    positions_ptr,
    col_indices_ptr,
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MurmurHash 32-bit implementation for Triton.
    Reference:
    - https://medium.com/@thealonemusk/murmurhash-the-scrappy-algorithm-that-secretly-powers-half-the-internet-2d3f79b4509b
    - https://en.wikipedia.org/wiki/MurmurHash

    We treat 64-bit seed, 32-bit position, and 32-bit col_index as 4 4-byte blocks, and bit-blend them together.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_idx = pid_row
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Load inputs
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)

    h: tl.uint32 = 0  # hash accumulator

    # Process seed_low
    k: tl.uint32 = (seed & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process seed_high
    k = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process position block starting from seed32
    h = murmur3_mix(h, pos)

    # Process col block
    h = murmur3_mix(h, col)

    # Finalize (len=16 for seed + pos + col)
    h ^= 16
    h = fmix32(h)

    # Store result as uint32
    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)


_U32 = 0xFFFFFFFF


def _murmur3_mix_torch(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # int64 math on values below 2**32: products wrap, but their low 32 bits are exact.
    k = (k * 0xCC9E2D51) & _U32
    k = ((k << 15) | (k >> 17)) & _U32
    k = (k * 0x1B873593) & _U32
    h = h ^ k
    h = ((h << 13) | (h >> 19)) & _U32
    return (h * 5 + 0xE6546B64) & _U32


def _murmur_hash32_torch(seed, positions, col_indices):
    """Pure-torch ``murmur_hash32`` for devices without Triton."""
    seed = seed.view(torch.int64) if seed.dtype == torch.uint64 else seed.long()
    h = _murmur3_mix_torch(torch.zeros_like(seed), seed & _U32)
    h = _murmur3_mix_torch(h, (seed >> 32) & _U32)
    h = _murmur3_mix_torch(h, positions.long() & _U32)
    h = _murmur3_mix_torch(h.unsqueeze(1), (col_indices.long() & _U32).unsqueeze(0))
    h = h ^ 16
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _U32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _U32
    h = h ^ (h >> 16)
    return h.to(torch.uint32)


def murmur_hash32(seed, positions, col_indices):
    assert seed.shape == positions.shape, (
        "Seed and positions must have the same shape (n,)"
    )
    assert len(seed.shape) == 1 and len(col_indices.shape) == 1, (
        f"Inputs must be 1D tensors {seed.shape=} {col_indices.shape=}"
    )
    from sglang.srt.platforms import current_platform

    if (
        current_platform.is_out_of_tree()
        and not current_platform.capabilities.supports_triton
    ):
        return _murmur_hash32_torch(seed, positions, col_indices)
    device = seed.device
    if device.type == "cpu":
        import sgl_kernel  # noqa: F401

        return torch.ops.sgl_kernel.murmur_hash32_cpu.default(
            seed, positions, col_indices
        )

    n = seed.shape[0]
    m = col_indices.shape[0]
    hashed = torch.empty((n, m), dtype=torch.uint32, device=device)

    BLOCK_SIZE = 1024
    grid = (n, triton.cdiv(m, BLOCK_SIZE))
    murmur_hash32_kernel[grid](
        seed, positions, col_indices, hashed, n, m, BLOCK_SIZE=BLOCK_SIZE
    )
    return hashed
