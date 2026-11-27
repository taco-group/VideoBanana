# Evaluation on ChatVE-Bench

A system is scored on every turn of every session: the checklist gives editing accuracy (EA, with its parts SA,
SPA, EP) and content preservation (CP); five video-quality metrics (TQ, VQR, PC, AES, MSM) are scored on every
output and on the source clips.

## 1. Setup

```bash
git clone https://github.com/taco-group/VideoBanana.git && cd VideoBanana
pip install -e ".[openai]"
hf download JiayiZhang14/ChatVE-Bench --repo-type dataset --local-dir ChatVE-Bench
```

## 2. Run your system

Edit every session turn by turn and save the output after each turn as one video with the source clip's frame
count (49 frames; the short clip of `mt_0031` has 41), for example `outputs/<session_id>/turn01.mp4`. The user's message at each turn is `request` in
`ChatVE-Bench/sessions.json`; the source clip is `ChatVE-Bench/clips/<session_id>.mp4`. If your editor needs
a longer input, cut its output back to the source clip's length.

## 3. Checklist (EA, CP)

The paper's judge is gemini-3.6-flash on Vertex AI, 8 frames per clip. Set `GOOGLE_CLOUD_PROJECT` (and
optionally `GOOGLE_CLOUD_LOCATION`, default `global`) and log in with
`gcloud auth application-default login`, then:

```bash
python -m chatve_bench.evaluate --bench ChatVE-Bench \
    --outputs "outputs/{session_id}/turn{turn:02d}.mp4" --out results/my_system
```

Any OpenAI-compatible server can be the judge instead (`--backend openai --base-url ... --model ...`), e.g. a
local model served by vLLM; its scores are not comparable with the paper's.

The judged answers are saved to `results/my_system/answers.jsonl` as they arrive, so an interrupted run resumes.
A question without a usable answer is asked again (`--retries`, default 2). `summary.md` reports SA, SPA, EP, EA
and CP for the final turn of every session and for all turns.

## 4. Video quality (TQ, VQR, PC, AES, MSM)

The metrics run with their original code, which is not part of this repository:

- [CoVEBench](https://github.com/NJU-LINK/CoVEBench): TQ (DOVER++), VQR (VisualQuality-R1), AES (aesthetic
  predictor v2.5) and MSM. Set up its environment and weights as in its README.
- [VideoPhy](https://github.com/Hritikbansal/videophy): PC with VideoCon-Physics
  (`videophysics/videocon_physics`). Set up its environment as in its README.

```bash
python -m chatve_bench.vq --bench ChatVE-Bench \
    --outputs "outputs/{session_id}/turn{turn:02d}.mp4" --out results/my_system \
    --covebench /path/to/CoVEBench --videophy /path/to/videophy \
    --videophy-python /path/to/videophy-env/bin/python --gpu 0
```

`vq_summary.md` reports the means over the final turns and over all turns, and the change of TQ and VQR from
each session's source clip (ΔTQ, ΔVQR). TQ and VQR sample at random, so repeated runs differ slightly.
