"""The CoinVE-Edit worker protocol, with a stand-in for the CoinVE-Edit repository and torch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from videobanana.editor import CoinVEEditor, CoinVESettings, EditorError, EditRequest

FAKE_TORCH = '''
class _Cuda:
    is_available = staticmethod(lambda: False)
    empty_cache = staticmethod(lambda: None)
cuda = _Cuda()
device = lambda name: name
'''

FAKE_ENTRYPOINT = '''
import argparse, json, os
print("noise on stdout from the model code")

def parse_args():
    p = argparse.ArgumentParser()
    for name in ("--src_video", "--composite_checkpoint", "--local_model_path", "--mllm_model", "--output_dir", "--output_name"):
        p.add_argument(name)
    p.add_argument("--prompts", nargs="+")
    for name in ("--eval_max_pixels", "--eval_max_frame", "--src_max_frames", "--seed", "--num_inference_steps"):
        p.add_argument(name, type=int)
    p.add_argument("--no_side_by_side", action="store_true")
    p.add_argument("--save_instruction_masks", action="store_true")
    p.add_argument("--residual_mode", default="replace_delta")
    return p.parse_args()

class _Value:
    def detach(self): return self
    float = cpu = detach
    def mean(self): return self
    def item(self): return 0.5

class _Masks:
    shape = (1,)
    def __getitem__(self, index): return _Value()

class _Pipe:
    device = "cpu"
    def load_models_to_device(self, names): pass

class _Residual:
    def clear(self): pass

def build_residual_inference_module(args, device):
    with open(os.path.join(os.getcwd(), "loads.txt"), "a") as f:
        f.write("load\\n")
    return None, _Pipe(), _Residual(), None

def read_src_pil_resized(path, frames, pixels):
    return ["frame"] * frames, None, None, 24.0

quantize_4kp1 = lambda n: (n - 1) // 4 * 4 + 1
predict_residual_contexts_and_masks = lambda pipe, head, trainer, prompts, frames, args, device: (", ".join(prompts), None, None, _Masks(), (1, 1, 1))
run_residual_with_pred_masks = lambda *args: args[6]
save_instruction_mask_videos = lambda *args: []

def save_video(frames, path, fps, quality):
    with open(path, "w") as f:
        json.dump({"frames": len(frames), "quality": quality}, f)
'''


def _settings(repo: Path, **overrides: str) -> CoinVESettings:
    options = dict(repo_root=str(repo), python=sys.executable, composite_checkpoint="ckpt", local_model_path="base", mllm_model="mllm")
    return CoinVESettings(**{**options, **overrides})


def test_the_worker_loads_once_and_answers_each_request(tmp_path: Path) -> None:
    repo = tmp_path / "CoinVE-Edit"
    repo.mkdir()
    (repo / "torch.py").write_text(FAKE_TORCH)
    (repo / "infer_coinve_single.py").write_text(FAKE_ENTRYPOINT)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"video")
    editor = CoinVEEditor(_settings(repo), tmp_path / "logs")
    try:
        for attempt in (1, 2):
            output = tmp_path / f"A{attempt}" / "output.mp4"
            editor.edit(EditRequest(input_video=str(source), prompts=["Add a hat."], seed=attempt - 1, frames=49, output_path=str(output)))
            assert output.read_text() == '{"frames": 49, "quality": 9}'
            assert (output.parent / "output_meta.json").is_file()
        assert (repo / "loads.txt").read_text() == "load\n"

        with pytest.raises(EditorError, match="FileNotFoundError"):
            editor.edit(EditRequest(input_video=str(tmp_path / "missing.mp4"), prompts=["x"], seed=0, frames=49, output_path=str(tmp_path / "o.mp4")))
    finally:
        editor.close()
    assert "noise on stdout from the model code" in (tmp_path / "logs" / "coinve_worker_gpu0.log").read_text()
