"""Judge backends. Each takes a system prompt and the parts of one request and returns the reply text.

- ``gemini``: Gemini on Vertex AI through the google-genai SDK. The paper uses gemini-3.6-flash, temperature 0,
  8 frames per clip. Set ``GOOGLE_CLOUD_PROJECT`` (and optionally ``GOOGLE_CLOUD_LOCATION``, default ``global``)
  and log in with Application Default Credentials.
- ``openai``: any OpenAI-compatible chat endpoint, e.g. a local vision-language model served by vLLM. Each clip
  is sent as evenly spaced frames (images), since such servers differ in how they accept videos. Results with
  another judge are not comparable with the paper's numbers.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2

MAX_OUTPUT_TOKENS = 65000
TEMPERATURE = 0.0


class GeminiBackend:
    def __init__(self, model: str = "gemini-3.6-flash", frames: int = 8, retries: int = 5):
        from google import genai  # imported here so the openai backend does not need it
        from google.genai import types

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError("set GOOGLE_CLOUD_PROJECT to the Google Cloud project that runs the Gemini judge")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self.client, self.types = genai.Client(vertexai=True, project=project, location=location), types
        self.model, self.frames, self.retries = model, frames, retries

    def _video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or count <= 0:
            raise ValueError(f"cannot read the frame rate or frame count of {path}")
        part = self.types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="video/mp4")
        # sample `frames` evenly spaced frames of the clip
        part.video_metadata = self.types.VideoMetadata(fps=min(24.0, fps, self.frames / (count / fps)))
        return part

    def generate(self, system_prompt: str, parts: list) -> str:
        contents = [self._video(v) if kind == "video" else v for kind, v in parts]
        config = self.types.GenerateContentConfig(system_instruction=system_prompt, temperature=TEMPERATURE,
                                                  max_output_tokens=MAX_OUTPUT_TOKENS)
        error = None
        for attempt in range(1, self.retries + 1):
            try:
                text = self.client.models.generate_content(model=self.model, contents=contents, config=config).text
                if text:
                    return text
                error = "empty reply"
            except Exception as e:  # noqa: BLE001 - network and quota errors are retried
                error = str(e)
            time.sleep(min(30, 5 * attempt))
        raise RuntimeError(error)


def sample_frames(path: Path, n: int, max_side: int = 640) -> list[str]:
    """``n`` evenly spaced frames of a clip as JPEG data URLs."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"cannot read {path}")
    picks = [frames[round(i * (len(frames) - 1) / max(1, n - 1))] for i in range(n)]
    urls = []
    for f in picks:
        h, w = f.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        urls.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", f)[1].tobytes()).decode())
    return urls


class OpenAIBackend:
    def __init__(self, model: str, base_url: str, frames: int = 8, retries: int = 5):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.frames, self.retries = model, frames, retries

    def generate(self, system_prompt: str, parts: list) -> str:
        content = []
        for kind, v in parts:
            if kind == "video":
                content += [{"type": "image_url", "image_url": {"url": u}} for u in sample_frames(v, self.frames)]
            else:
                content.append({"type": "text", "text": v})
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        error = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=TEMPERATURE)
                text = r.choices[0].message.content
                if text:
                    return text
                error = "empty reply"
            except Exception as e:  # noqa: BLE001
                error = str(e)
            time.sleep(min(30, 5 * attempt))
        raise RuntimeError(error)
