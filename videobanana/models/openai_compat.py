"""Any OpenAI-compatible chat endpoint, e.g. an open model served by vLLM (the paper's Qwen variant).

The endpoint receives no video parts: a call with videos sends the frames
Gemini would sample at ``media_fps`` (evenly spaced, at most 1280 px wide,
JPEG) as images, after a "Video k (n frames in time order):" label. Several
server URLs spread the calls; each goes to the least busy one.

A retry uses the next seed (the same seed would repeat an answer cut off by
the token limit) and, once the prompt length is known, all the room the
context leaves. The last retry answers without thinking, for the rare call
whose reasoning fills every token.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import JsonResponse, parse_json_response

FRAME_WIDTH = 1280


def sample_frames_b64(video: str, fps: float | None, width: int = FRAME_WIDTH) -> list[str]:
    """JPEG frames of ``video`` at ``fps`` (8 frames without one), base64-encoded."""
    import cv2

    cap = cv2.VideoCapture(video)
    native = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    count = max(1, round(len(frames) / native * fps)) if fps else min(len(frames), 8)
    out = []
    for index in range(count):
        frame = frames[min(len(frames) - 1, int(index * len(frames) / count))]
        height, frame_width = frame.shape[:2]
        if frame_width > width:
            frame = cv2.resize(frame, (width, round(height * width / frame_width)), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(base64.b64encode(buffer.tobytes()).decode())
    return out


class OpenAICompatClient:
    def __init__(
        self,
        base_urls: list[str] | str,
        *,
        max_tokens: int = 16384,
        context_len: int = 40960,
        timeout: float = 1800.0,
        attempts: int = 3,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        if isinstance(base_urls, str):  # one string, comma-separated
            base_urls = base_urls.split(",")
        self.urls = [url.strip().rstrip("/") for url in base_urls if url.strip()]
        if not self.urls:
            raise ValueError("no model server URL")
        self.busy = {url: 0 for url in self.urls}
        self.lock = threading.Lock()
        self.max_tokens, self.context_len, self.timeout, self.attempts = max_tokens, context_len, timeout, attempts
        self.api_key = os.environ.get(api_key_env, "")

    def preflight(self) -> list[str]:
        errors = []
        for url in self.urls:
            try:
                urllib.request.urlopen(urllib.request.Request(f"{url}/models", headers=self._headers()), timeout=20)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                errors.append(f"model server {url} not reachable: {exc}")
        return errors

    def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        media_paths: list[str] | None = None,
        media_fps: float | None = None,
    ) -> JsonResponse:
        content: Any = prompt
        if media_paths:
            content = [{"type": "text", "text": prompt}]
            for number, path in enumerate(media_paths, start=1):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"media input missing: {path}")
                frames = sample_frames_b64(path, media_fps)
                content.append({"type": "text", "text": f"Video {number} ({len(frames)} frames in time order):"})
                content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item}"}} for item in frames]
        messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": content}]
        last: Exception | None = None
        prompt_tokens = 0
        for attempt in range(self.attempts):
            body: dict[str, Any] = {
                "model": model,
                "seed": attempt,
                "messages": messages,
                "max_tokens": max(self.max_tokens, self.context_len - prompt_tokens - 64) if prompt_tokens else self.max_tokens,
            }
            last_try = attempt == self.attempts - 1 and attempt > 0
            if last_try:
                body["chat_template_kwargs"] = {"enable_thinking": False}
                print(f"[openai_compat] last attempt without thinking for a {model} call: {last}", file=sys.stderr, flush=True)
            with self.lock:
                url = min(self.urls, key=lambda item: self.busy[item])
                self.busy[url] += 1
            try:
                request = urllib.request.Request(
                    f"{url}/chat/completions", data=json.dumps(body).encode(), headers=self._headers()
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                    out = json.loads(answer.read())
                prompt_tokens = (out.get("usage") or {}).get("prompt_tokens") or prompt_tokens
                raw_text = str(out["choices"][0]["message"].get("content") or "").strip()
                if not raw_text:
                    raise RuntimeError(f"empty answer (finish_reason {out['choices'][0].get('finish_reason')})")
                # a server without a reasoning parser returns the reasoning inline: parse what follows it
                answer_text = raw_text.rsplit("</think>", 1)[-1].strip() or raw_text
                return JsonResponse(raw_text=raw_text, payload=parse_json_response(answer_text), model=model)
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last = exc
                time.sleep(min(60.0, 5.0 * 2**attempt))
            finally:
                with self.lock:
                    self.busy[url] -= 1
        raise RuntimeError(f"model call failed after {self.attempts} attempts: {last}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
