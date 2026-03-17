import random
import numpy as np
import torch

from pfns.experiments.model_benchmarks.sampling import ClassCoverageBatchGenerator

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

gen = ClassCoverageBatchGenerator(
    num_batches=5,
    smallest_seqlen=250,
    largest_seqlen=128_000,  # change if needed
    num_features=10,
    num_classes=5,
    number_of_test_samples=100,
    prior_device="cpu",      # keep fixed for comparison
)

for rep, (batch, gen_ms) in enumerate(gen, start=1):
    print(f"\n=== batch {rep} | gen_ms={gen_ms:.2f} ===")

    print("x first 10:", batch.x.flatten()[:10].cpu())
    print("y first 10:", batch.y.flatten()[:10].cpu())
    print("target_y first 10:", batch.target_y.flatten()[:10].cpu())
    print("categorical_mask first 10:", batch.categorical_mask.flatten()[:10].cpu())
