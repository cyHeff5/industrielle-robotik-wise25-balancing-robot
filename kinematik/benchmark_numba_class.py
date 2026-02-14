import argparse
import statistics
import time

import numpy as np

from numba_abstande import NumbaAbstande


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark for NumbaAbstande class.")
    parser.add_argument("--n", type=int, default=200_000, help="Number of function evaluations per run.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    s_r = np.array([1.0, -0.4, 0.9], dtype=np.float64)
    s_v = np.array([0.8, 0.6, -0.7], dtype=np.float64)
    s_l = np.array([-0.5, 0.3, 1.1], dtype=np.float64)

    solver = NumbaAbstande(s_r=s_r, s_v=s_v, s_l=s_l, l=216.5064)
    solver.warmup()

    inputs = [rng.uniform(50.0, 350.0, size=3).astype(np.float64) for _ in range(args.n)]

    times = []
    checksum = 0.0
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        acc = 0.0
        for a in inputs:
            out = solver.evaluate(a)
            acc += float(out[0] + out[1] + out[2])
        dt = time.perf_counter() - t0
        times.append(dt)
        checksum = acc

    print(f"Benchmark NumbaAbstande: n={args.n}, repeats={args.repeats}")
    print(", ".join(f"run{i + 1}={t:.6f}s" for i, t in enumerate(times)))
    print("per_call " + ", ".join(f"run{i + 1}={(t / args.n) * 1e6:.3f}us" for i, t in enumerate(times)))
    print(
        f"min={min(times):.6f}s mean={statistics.mean(times):.6f}s std={statistics.pstdev(times):.6f}s "
        f"| mean_per_call={(statistics.mean(times) / args.n) * 1e6:.3f}us checksum={checksum:.6f}"
    )


if __name__ == "__main__":
    main()
