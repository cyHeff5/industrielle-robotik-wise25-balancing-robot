
import argparse
import time

import numpy as np

from vision.ball_detection.runtime import BallDetectorRuntime
from vision.ball_detection.types import HsvRange


def _print_stats(name: str, values_ms: list[float]) -> None:
    arr = np.asarray(values_ms, dtype=np.float64)
    print(
        f"{name:>10}: "
        f"mean={arr.mean():7.3f} ms  "
        f"median={np.median(arr):7.3f} ms  "
        f"p95={np.percentile(arr, 95):7.3f} ms  "
        f"p99={np.percentile(arr, 99):7.3f} ms  "
        f"min={arr.min():7.3f} ms  "
        f"max={arr.max():7.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure camera-to-ball-position latency on Raspberry Pi."
    )
    parser.add_argument("--frames",        type=int,   default=500)
    parser.add_argument("--warmup",        type=int,   default=50)
    parser.add_argument("--width",         type=int,   default=2592)
    parser.add_argument("--height",        type=int,   default=2592)
    parser.add_argument("--input-color-order", choices=("RGB", "BGR"), default="BGR")
    parser.add_argument("--use-roi",       action="store_true")
    parser.add_argument("--roi-size",      type=int,   default=700)
    parser.add_argument("--roi-miss-limit",type=int,   default=5)
    parser.add_argument("--downsample",    type=int,   default=1)
    parser.add_argument("--print-every",   type=int,   default=50)
    args = parser.parse_args()

    hsv_range = HsvRange(
        lower=np.array([90, 80, 60],    dtype=np.uint8),
        upper=np.array([130, 255, 255], dtype=np.uint8),
    )

    runtime = BallDetectorRuntime(
        hsv_range=hsv_range,
        frame_size=(args.width, args.height),
        input_color_order=args.input_color_order,
        use_roi_tracking=args.use_roi,
        roi_size=args.roi_size,
        roi_miss_limit=args.roi_miss_limit,
        downsample_factor=args.downsample,
    )
    runtime.start()

    # Warten bis Background-Thread den ersten Frame geliefert hat
    while runtime.get_latest_frame() is None:
        time.sleep(0.01)

    print(
        f"Starting latency measurement: warmup={args.warmup}, measured={args.frames}, "
        f"resolution={args.width}x{args.height}, input_color_order={args.input_color_order}, "
        f"use_roi={args.use_roi}, downsample={args.downsample}"
    )

    get_frame_ms: list[float] = []
    detect_ms:    list[float] = []
    total_ms:     list[float] = []
    valid_count = 0

    try:
        for i in range(args.warmup + args.frames):
            t0 = time.perf_counter()
            frame_bgr = runtime.get_latest_frame()
            t1 = time.perf_counter()
            measurement, _ = runtime.read_with_frame()
            t2 = time.perf_counter()

            if i < args.warmup:
                continue

            gf    = (t1 - t0) * 1000.0
            det   = (t2 - t1) * 1000.0
            total = (t2 - t0) * 1000.0

            get_frame_ms.append(gf)
            detect_ms.append(det)
            total_ms.append(total)
            valid_count += int(measurement.valid)

            measured_idx = i - args.warmup + 1
            if args.print_every > 0 and measured_idx % args.print_every == 0:
                print(
                    f"[{measured_idx:4d}/{args.frames}] "
                    f"total={total:7.3f} ms  get_frame={gf:7.3f} ms  "
                    f"detect={det:7.3f} ms  valid={measurement.valid}"
                )
    finally:
        runtime.stop()

    if not total_ms:
        print("No frames measured.")
        return

    print("\nLatency summary:")
    _print_stats("get_frame", get_frame_ms)
    _print_stats("detect",    detect_ms)
    _print_stats("total",     total_ms)

    mean_total = float(np.mean(total_ms))
    est_fps = 1000.0 / mean_total if mean_total > 0.0 else 0.0
    valid_ratio = (valid_count / len(total_ms)) * 100.0
    print(f"\nEstimated FPS (from mean total): {est_fps:.2f}")
    print(f"Detection valid ratio: {valid_ratio:.1f}%")


if __name__ == "__main__":
    main()
