"""
SDTM Benchmark (Paper Table 1 reproduction)

Paper: "Attend to Not Attended: Structure-then-Detail Token Merging
        for Post-training DiT Acceleration"

MACs measurement: Hook-based measurement during actual inference
"""

import torch
import time
import json
import argparse
from pathlib import Path
from tabulate import tabulate


def measure_macs_with_hooks(pipe, prompt, device, num_steps=50, seed=42):
    """Measure MACs during actual inference using forward hooks.

    This method accurately measures MACs for token merging methods (ToMe, SDTM)
    by tracking actual tensor sizes during inference.

    Args:
        pipe: The diffusion pipeline
        prompt: Text prompt for inference
        device: Device to run on
        num_steps: Number of inference steps
        seed: Random seed

    Returns:
        Average MACs per step (to match paper's metric)
    """
    total_macs = [0]  # Use list to allow modification in nested function
    hooks = []

    def linear_hook(module, input, output):
        """Calculate MACs for Linear layer: tokens * in_features * out_features"""
        if len(input) > 0 and input[0] is not None:
            inp = input[0]
            if inp.dim() == 3:
                batch, seq_len, in_features = inp.shape
                out_features = module.out_features
                macs = batch * seq_len * in_features * out_features
            elif inp.dim() == 2:
                batch, in_features = inp.shape
                out_features = module.out_features
                macs = batch * in_features * out_features
            else:
                macs = 0
            total_macs[0] += macs

    # Register hooks on all Linear layers in transformer
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(linear_hook)
            hooks.append(hook)

    # Run full inference to measure total MACs
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.0,
            generator=generator,
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Return average MACs per step (matches paper's metric)
    return total_macs[0] / num_steps


def run_inference(pipe, prompt, steps, seed, device, save_path=None):
    """Run inference and measure latency."""
    generator = torch.Generator(device=device).manual_seed(seed)

    torch.cuda.synchronize()
    start = time.perf_counter()

    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=7.0,
        generator=generator,
    )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    if save_path:
        result.images[0].save(save_path)
        print(f"  Saved: {save_path}")

    return latency


def main():
    parser = argparse.ArgumentParser(description="SDTM Benchmark")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "sdtm", "tome"])
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--deviation", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str,
                        default="An Asian family getting together for an enjoyable Chinese dinner")
                        # default="a smiling golden retriever")
    parser.add_argument("--output_dir", type=str, default="./benchmark_outputs")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--merge_mlp", type=int, default=1, help="ToMe: merge MLP (1=True, 0=False)")
    args = parser.parse_args()

    # Paths
    cache_dir = "/home/server39/donghyeon_workspace/donghyeon_mount/huggingface_cache_dir"
    model_path = f"{cache_dir}/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Ratio: {args.ratio}, Deviation: {args.deviation}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")
    print("=" * 60)

    torch.cuda.empty_cache()

    # Load model
    from diffusers import StableDiffusion3Pipeline
    print("\nLoading SD3 Medium...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe = pipe.to(args.device)

    tore_info = None

    # Apply SDTM if needed
    if args.method == "sdtm":
        from TR_SDTM import apply_SDTM

        switch_step = int(args.steps * 0.4)

        apply_SDTM(
            pipe,
            ratio=args.ratio,
            deviation=args.deviation,
            switch_step=switch_step,
        )
        print(f"\nApplied SDTM (ratio={args.ratio}, deviation={args.deviation}, switch_step={switch_step})")
        tore_info = pipe._tore_info

    # Apply ToMe if needed
    elif args.method == "tome":
        from TR_ToMe import apply_ToMe

        merge_mlp = bool(args.merge_mlp)
        apply_ToMe(
            pipe,
            ratio=args.ratio,
            merge_attn=True,
            merge_mlp=merge_mlp,
        )
        print(f"\nApplied ToMe (ratio={args.ratio}, merge_mlp={merge_mlp})")
        tore_info = pipe._tore_info

    # Measure MACs with hooks during actual inference (also serves as warmup)
    print("\nMeasuring MACs with hooks (warmup run)...")
    macs = measure_macs_with_hooks(pipe, args.prompt, args.device, args.steps, args.seed)
    macs_t = macs / 1e12
    print(f"MACs (measured): {macs_t:.2f}T")

    # Benchmark
    print(f"\nBenchmark ({args.num_runs} runs):")
    latencies = []
    for i in range(args.num_runs):
        if i == 0:
            if args.method == "baseline":
                save_path = str(output_dir / "baseline.png")
            elif args.method == "sdtm":
                save_path = str(output_dir / f"sdtm_r{args.ratio}_d{args.deviation}.png")
            else:  # tome
                save_path = str(output_dir / f"tome_r{args.ratio}.png")
        else:
            save_path = None
        lat = run_inference(pipe, args.prompt, args.steps, args.seed, args.device, save_path)
        latencies.append(lat)
        print(f"  Run {i+1}: {lat:.2f}s")

    avg_latency = sum(latencies) / len(latencies)
    std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

    # Save result
    result = {
        "method": args.method,
        "ratio": args.ratio,
        "deviation": args.deviation,
        "macs_t": round(macs_t, 2),
        "latency_s": round(avg_latency, 2),
        "latency_std": round(std_latency, 2),
        "steps": args.steps,
        "seed": args.seed,
    }

    if args.method == "baseline":
        result_file = output_dir / "baseline_result.json"
    elif args.method == "sdtm":
        result_file = output_dir / f"sdtm_r{args.ratio}_d{args.deviation}_result.json"
    else:  # tome
        result_file = output_dir / f"tome_r{args.ratio}_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"MACs: {macs_t:.2f}T, Latency: {avg_latency:.2f}s (±{std_latency:.2f})")
    print(f"Result saved: {result_file}")
    print("=" * 60)

    del pipe
    torch.cuda.empty_cache()


def aggregate_results(output_dir="./benchmark_outputs"):
    """Aggregate results and compute speed ratios."""
    output_dir = Path(output_dir)
    results = []
    baseline_latency = None

    # Load baseline first
    baseline_file = output_dir / "baseline_result.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            data = json.load(f)
            data["speed"] = 1.0
            baseline_latency = data["latency_s"]
            results.append(data)

    # Load SDTM results (sorted by ratio)
    sdtm_files = sorted(output_dir.glob("sdtm_r*_d*_result.json"))
    for path in sdtm_files:
        with open(path) as f:
            data = json.load(f)
        if baseline_latency:
            data["speed"] = round(baseline_latency / data["latency_s"], 2)
        results.append(data)

    # Load ToMe results (sorted by ratio)
    tome_files = sorted(output_dir.glob("tome_r*_result.json"))
    for path in tome_files:
        with open(path) as f:
            data = json.load(f)
        if baseline_latency:
            data["speed"] = round(baseline_latency / data["latency_s"], 2)
        results.append(data)

    if not results:
        print("No results found.")
        return

    # Print comparison table
    print("\n" + "=" * 70)
    print("SDTM Benchmark Results vs Paper SDTM*-a")
    print("=" * 70)

    headers = ["Method", "Ratio", "Dev", "MACs(T)", "Latency(s)", "Speed"]
    rows = []
    for r in results:
        method_name = r["method"].upper()
        if r["method"] == "sdtm":
            method_name = "SDTM"
        elif r["method"] == "tome":
            method_name = "ToMe"
        rows.append([
            method_name,
            "-" if r["method"] == "baseline" else f"{r['ratio']:.1f}",
            "-" if r["method"] == "baseline" or r["method"] == "tome" else f"{r['deviation']:.1f}",
            f"{r['macs_t']:.2f}",
            f"{r['latency_s']:.2f}",
            f"{r.get('speed', '-'):.2f}x"
        ])

    # Add paper reference
    rows.append(["---", "---", "---", "---", "---", "---"])
    rows.append(["Paper Baseline", "-", "-", "6.01", "10.67*", "1.00x"])
    rows.append(["Paper SDTM*-a", "?", "?", "4.20", "8.20*", "1.30x"])

    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print("\n* Paper uses 4×A100, we use 1×RTX4090")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    import sys
    if "--aggregate" in sys.argv:
        # Parse output_dir for aggregate
        output_dir = "./benchmark_outputs"
        for i, arg in enumerate(sys.argv):
            if arg == "--output_dir" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
        aggregate_results(output_dir)
    else:
        main()
