# Running VideoBanana

VideoBanana edits a ChatVE-Bench session turn by turn. At every turn the intent refiner restates the goal of
the conversation, the entry selector picks the checkpoint to edit from, the instruction writer turns the
missing intents into editor instructions, the intent compiler checks them (it blocks an ambiguous
instruction, and that attempt makes no edit), CoinVE-Edit makes the edit, and the verifier checks the output
against the whole goal. A turn makes at most two attempts (seeds 0 and 1) and commits the first one that
passes, or else the better-scored one. If the entry checkpoint already meets the goal, as after most undos,
the turn returns it without editing.

## 1. Install

```bash
git clone https://github.com/taco-group/VideoBanana.git && cd VideoBanana
pip install -e .                    # add ".[batch]" for Gemini batch mode
hf download JiayiZhang14/ChatVE-Bench --repo-type dataset --local-dir ChatVE-Bench
```

## 2. Set up CoinVE-Edit

VideoBanana calls [CoinVE-Edit](https://github.com/coinve200k/CoinVE-200K) unchanged, in its own Python
environment. The paper uses commit `ce091d4f7cc13af09ed276600b0b84db779c915d`.

```bash
git clone https://github.com/coinve200k/CoinVE-200K.git && cd CoinVE-200K
git checkout ce091d4f7cc13af09ed276600b0b84db779c915d
```

Create its environment and download its weights as described in its README (the composite checkpoint and
the base models, including Qwen3-VL-8B-Instruct). Then set the paths in the `editor` block of your config:

| Key | Value |
|---|---|
| `repo_root` | the `CoinVE-Edit` folder of the clone |
| `python` | the Python of the CoinVE-Edit environment |
| `composite_checkpoint` | `coinve_edit_composite_vllm256_dit128.safetensors` |
| `local_model_path` | the base model folder |
| `mllm_model` | the Qwen3-VL-8B-Instruct folder |
| `gpus` | GPU ids; one CoinVE-Edit worker per GPU |

Each worker loads the model once and keeps it on its GPU. One edit of a 49-frame 720p clip fits on a 48 GB GPU.
A worker that gives no answer within `job_timeout` seconds (default 3600) is killed and restarted.

## 3. Choose the models

**Gemini (as in the paper): `configs/gemini.yaml`.** The refiner and writer use gemini-3.6-flash and the
verifier gemini-3.7-flash, which sees 8 frames of each video.

- Vertex AI, online (the default): set `GOOGLE_CLOUD_PROJECT` and log in with
  `gcloud auth application-default login`.
- Vertex AI, batch mode (half price, but every call waits in the batch queue, so a run takes longer): set
  `mode: batch` and `batch_bucket` to a Cloud Storage bucket you can write to. Requests and videos are staged
  there.

**An open model: `configs/qwen.yaml`.** Every model role uses Qwen3.8-27B served by vLLM (the command is at
the top of the file). Its verifier passes an intent only at the top score (`item_gate_floor: 3`), as in the
paper. List several servers in `base_urls` to share the calls.

## 4. Run

```bash
python -m videobanana.run --bench ChatVE-Bench --out runs/videobanana --config configs/gemini.yaml
```

- `--sessions mt_0001 mt_0002` runs only these sessions.
- `--workers N` runs N sessions in parallel (default 8). The turns of a session always run in order; edits
  wait for a free GPU worker.
- Re-running the same command resumes: finished turns are skipped and an interrupted turn continues where it
  stopped. A session that fails is logged to `runs/videobanana/failures.log` and the others go on.

Outputs:

```
runs/videobanana/
  manifest.json              models, settings and prompt versions of the run
  logs/                      CoinVE-Edit worker logs
  <session_id>/
    turn01.mp4 ...           the video after each turn (49 frames)
    ...                      the conversation state: intents, checkpoints, instructions, edits, verdicts
```

## 5. Evaluate

```bash
python -m chatve_bench.evaluate --bench ChatVE-Bench \
    --outputs "runs/videobanana/{session_id}/turn{turn:02d}.mp4" --out results/videobanana
```

See [evaluation.md](evaluation.md) for the judge and the video-quality metrics. Model outputs and video
generation are sampled, so a new run will not reproduce the paper's numbers exactly.
