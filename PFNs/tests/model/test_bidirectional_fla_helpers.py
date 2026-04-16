import torch

from pfns.model.bidirectional_fla import (
    BidirectionalFLACache,
    FusedBidirectionalFLACache,
    fuse_bidirectional_cache,
)
from pfns.model.fla_cache_utils import copy_cache, repeat_cache


def test_bidirectional_cache_copy_and_repeat_use_wrapper_logic() -> None:
    cache = BidirectionalFLACache(
        forward_cache=torch.randn(2, 3),
        backward_cache=torch.randn(2, 3),
    )

    copied = copy_cache(cache)
    repeated = repeat_cache(cache, 2)

    assert isinstance(copied, BidirectionalFLACache)
    assert copied is not cache
    torch.testing.assert_close(copied.forward_cache, cache.forward_cache)
    torch.testing.assert_close(copied.backward_cache, cache.backward_cache)

    assert isinstance(repeated, BidirectionalFLACache)
    torch.testing.assert_close(
        repeated.forward_cache,
        cache.forward_cache.repeat_interleave(2, dim=0),
    )
    torch.testing.assert_close(
        repeated.backward_cache,
        cache.backward_cache.repeat_interleave(2, dim=0),
    )


def test_fused_bidirectional_cache_copy_and_repeat_use_wrapper_logic() -> None:
    cache = FusedBidirectionalFLACache(
        cache=torch.randn(2, 3),
        state_fusion="mean_output_mean_cache",
    )

    copied = copy_cache(cache)
    repeated = repeat_cache(cache, 2)

    assert isinstance(copied, FusedBidirectionalFLACache)
    assert copied.state_fusion == cache.state_fusion
    torch.testing.assert_close(copied.cache, cache.cache)

    assert isinstance(repeated, FusedBidirectionalFLACache)
    assert repeated.state_fusion == cache.state_fusion
    torch.testing.assert_close(
        repeated.cache,
        cache.cache.repeat_interleave(2, dim=0),
    )


def test_fuse_bidirectional_cache_averages_cache_tensors() -> None:
    cache = BidirectionalFLACache(
        forward_cache=torch.tensor([[1.0, 3.0]]),
        backward_cache=torch.tensor([[3.0, 5.0]]),
    )

    fused = fuse_bidirectional_cache(
        cache,
        state_fusion="mean_output_mean_cache",
    )

    torch.testing.assert_close(fused, torch.tensor([[2.0, 4.0]]))
