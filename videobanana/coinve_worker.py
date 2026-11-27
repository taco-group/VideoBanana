#!/usr/bin/env python3
"""Persistent CoinVE-Edit worker: loads the model once and edits videos on request.

Runs inside the CoinVE-Edit environment (it imports the repository's
``infer_coinve_single.py``), started by ``videobanana.editor.CoinVEEditor`` with
the repository as working directory and one GPU visible.

Protocol, one JSON object per line:

- stdin: ``{"argv": [...]}`` with the arguments of ``infer_coinve_single.py``;
  the model-loading arguments must stay the same across requests. The worker
  exits at the end of its input.
- stdout: ``{"status": "success", "output_path": ..., "elapsed_seconds": ...}``
  or ``{"status": "error", "error": ...}`` for each request.

Everything else printed while running goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: Arguments that change per request; all others load the model.
JOB_ARGUMENTS = {
    "src_video",
    "prompts",
    "output_dir",
    "output_name",
    "eval_max_pixels",
    "eval_max_frame",
    "src_max_frames",
    "seed",
    "num_inference_steps",
    "save_instruction_masks",
    "no_side_by_side",
    "show_progress",
}
#: imageio quality of the saved edit (as CoinVE-Edit's benchmark script saves its outputs).
OUTPUT_QUALITY = 9


def parse_job(single: Any, entrypoint: Path, argv: list[str]) -> argparse.Namespace:
    previous = sys.argv
    try:
        sys.argv = [str(entrypoint), *argv]
        return single.parse_args()
    finally:
        sys.argv = previous


def run_job(single: Any, model: tuple[Any, Any, Any, Any, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Edit one video and save it with its metadata and per-instruction mask videos."""
    torch, trainer, pipe, residual_module, mask_head = model
    if not os.path.isfile(args.src_video):
        raise FileNotFoundError(args.src_video)
    os.makedirs(args.output_dir, exist_ok=True)
    prompts = [str(prompt).strip() for prompt in args.prompts if str(prompt).strip()]
    if not prompts:
        raise ValueError("at least one --prompts instruction is required")
    out_name = args.output_name or f"{Path(args.src_video).stem}_edited"
    if not out_name.endswith(".mp4"):
        out_name += ".mp4"
    out_path = os.path.join(args.output_dir, out_name)
    print(f"[worker] src={args.src_video} prompts={prompts} out={out_path}", flush=True)

    started = time.time()
    try:
        frame_limits = [value for value in (args.eval_max_frame, args.src_max_frames) if value is not None and value > 0]
        src_frames, _, _, src_fps = single.read_src_pil_resized(
            args.src_video, min(frame_limits) if frame_limits else None, args.eval_max_pixels
        )
        source_count = min(len(src_frames), args.src_max_frames) if args.src_max_frames else len(src_frames)
        dit_frames = single.quantize_4kp1(source_count)
        if dit_frames < 5:
            raise RuntimeError(f"source video too short after quantization: t_dit={dit_frames}")
        src_frames = src_frames[:dit_frames]

        global_prompt, global_context, local_contexts, masks, grid = single.predict_residual_contexts_and_masks(
            pipe, mask_head, trainer, prompts, src_frames, args, pipe.device
        )
        video = single.run_residual_with_pred_masks(
            pipe, residual_module, global_prompt, global_context, local_contexts, masks, src_frames, args, pipe.device
        )
        edited = [frame.convert("RGB") if hasattr(frame, "convert") else frame for frame in video]
        single.save_video(edited, out_path, fps=src_fps if src_fps > 0 else 15, quality=OUTPUT_QUALITY)
        mask_paths: list[str] = []
        if args.save_instruction_masks:
            mask_paths = single.save_instruction_mask_videos(
                src_frames, masks, os.path.join(args.output_dir, "instruction_masks"), Path(out_name).stem, src_fps
            )
        elapsed = time.time() - started
        meta = {
            "src": args.src_video,
            "prompts": prompts,
            "global_prompt": global_prompt,
            "K": len(prompts),
            "grid_thw": list(grid),
            "mask_means": [float(masks[index].detach().float().mean().cpu().item()) for index in range(masks.shape[0])],
            "residual_mode": args.residual_mode,
            "eval_max_pixels": args.eval_max_pixels,
            "eval_max_frame": args.eval_max_frame,
            "seed": args.seed,
            "out": out_path,
            "mask_videos": mask_paths,
            "time": elapsed,
        }
        with open(os.path.join(args.output_dir, f"{Path(out_name).stem}_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        print(f"[worker] saved {out_path} time={elapsed:.2f}s", flush=True)
        return {"status": "success", "output_path": out_path, "elapsed_seconds": elapsed}
    finally:
        residual_module.clear()
        torch.cuda.empty_cache()
        # The next job starts with context and mask prediction: keep the MLLM on the GPU.
        try:
            pipe.load_models_to_device(("mllm",))
        except Exception:  # noqa: BLE001 - only a placement hint
            traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrypoint", required=True, help="path of infer_coinve_single.py")
    options = parser.parse_args()

    # Answers go to the original stdout; anything else written there (by this
    # script, the model code or native libraries) goes to stderr.
    sys.stdout.flush()
    answers = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)

    entrypoint = Path(options.entrypoint).resolve()
    sys.path.insert(0, str(entrypoint.parent))
    import torch
    import infer_coinve_single as single

    model: tuple[Any, Any, Any, Any, Any] | None = None
    signature: dict[str, Any] | None = None
    print(f"[worker] pid={os.getpid()} entrypoint={entrypoint}", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            args = parse_job(single, entrypoint, list(json.loads(line)["argv"]))
            current = {key: value for key, value in vars(args).items() if key not in JOB_ARGUMENTS}
            if model is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                trainer, pipe, residual_module, mask_head = single.build_residual_inference_module(args, device)
                model, signature = (torch, trainer, pipe, residual_module, mask_head), current
            elif current != signature:
                raise RuntimeError("model-loading arguments changed; start a new worker")
            result = run_job(single, model, args)
        except Exception as exc:  # noqa: BLE001 - the worker stays up for the next request
            traceback.print_exc()
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        answers.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
