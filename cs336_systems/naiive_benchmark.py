import argparse
import json
import sys
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
CS336_BASICS_ROOT = REPO_ROOT / "cs336-basics"
DEFAULT_PRESET_FILE = Path(__file__).with_name("model_size_presets.json")
if str(CS336_BASICS_ROOT) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS_ROOT))


def load_presets(path: Path) -> dict[str, dict[str, int]]:
    with path.open() as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark forward and backward passes for BasicsTransformerLM."
    )
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET_FILE)
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Named size preset from the preset JSON, for example small or medium.",
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="Print the available named size presets and exit.",
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    presets = load_presets(args.preset_file)
    if args.list_sizes:
        for name, config in presets.items():
            print(
                f"{name}: d_model={config['d_model']}, d_ff={config['d_ff']}, "
                f"num_layers={config['num_layers']}, num_heads={config['num_heads']}"
            )
        raise SystemExit(0)

    if args.size is not None:
        if args.size not in presets:
            available = ", ".join(presets)
            raise SystemExit(
                f"Unknown size '{args.size}'. Available sizes: {available}"
            )
        preset = presets[args.size]
        args.d_model = preset["d_model"]
        args.d_ff = preset["d_ff"]
        args.num_layers = preset["num_layers"]
        args.num_heads = preset["num_heads"]

    return args


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(args, device: torch.device):
    from cs336_basics.model import BasicsTransformerLM

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    return model.to(device)


def generate_batch(args, device: torch.device):
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    y = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    return x, y


def run_forward_backward_step(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)

    forward_start = default_timer()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    maybe_synchronize(device)
    forward_time = default_timer() - forward_start

    backward_start = default_timer()
    loss.backward()
    maybe_synchronize(device)
    backward_time = default_timer() - backward_start

    return forward_time, backward_time


def summarize_times(name: str, times: list[float]) -> tuple[float, float]:
    times_tensor = torch.tensor(times, dtype=torch.float64)
    mean_time_s = times_tensor.mean().item()
    median_time_s = times_tensor.median().item()
    print(f"mean {name} time: {mean_time_s * 1000:.2f} ms")
    print(f"median {name} time: {median_time_s * 1000:.2f} ms")
    return mean_time_s, median_time_s


def benchmark(args) -> None:
    device = torch.device(args.device)
    model = build_model(args, device)
    x, y = generate_batch(args, device)
    model.train()

    for _ in range(args.warmup_steps):
        run_forward_backward_step(model, x, y, device)

    forward_times = []
    backward_times = []
    total_times = []

    for _ in range(args.steps):
        step_start = default_timer()
        forward_time, backward_time = run_forward_backward_step(model, x, y, device)
        total_times.append(default_timer() - step_start)
        forward_times.append(forward_time)
        backward_times.append(backward_time)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = args.batch_size * args.context_length

    print(f"device: {device}")
    print(f"size preset: {args.size if args.size is not None else 'custom'}")
    print(f"parameters: {num_params / 1e6:.2f}M")
    print(
        f"model config: d_model={args.d_model}, d_ff={args.d_ff}, "
        f"num_layers={args.num_layers}, num_heads={args.num_heads}"
    )
    print(f"batch shape: ({args.batch_size}, {args.context_length})")
    print(f"warmup steps: {args.warmup_steps}")
    print(f"timed steps: {args.steps}")

    forward_mean_time_s, _ = summarize_times("forward", forward_times)
    summarize_times("backward", backward_times)
    summarize_times("total step", total_times)
    print(
        f"mean throughput: {tokens_per_step / forward_mean_time_s:.2f} forward tokens/s"
    )


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
