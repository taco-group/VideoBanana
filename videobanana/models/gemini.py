"""Gemini on Vertex AI through google-genai, online or as batch jobs.

Online calls send the videos inline. Batch jobs cost half as
much but take their requests and videos from a Cloud Storage bucket and wait in
a queue, so they pay off when many sessions run at once (``--workers``): the
requests that arrive within a short window share one job, and each caller
blocks until its own answer is back.

Every call asks for a JSON answer at temperature 0.1 with up to 8192 output
tokens, and samples each video at ``media_fps`` frames per second.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import JsonResponse, parse_json_response

LOG = logging.getLogger(__name__)

#: Highest video sampling rate the service accepts.
MAX_MEDIA_FPS = 24.0


@dataclass(frozen=True)
class GeminiSettings:
    #: Google Cloud project; credentials come from Application Default Credentials.
    project: str | None = None
    location: str = "global"
    #: ``online`` or ``batch``.
    mode: str = "online"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    max_retries: int = 3
    #: Batch mode: bucket (``gs://name``) where requests, videos and answers are staged.
    batch_bucket: str = ""
    batch_prefix: str = "videobanana-batch"
    #: A job is submitted once no request arrived for ``batch_idle_s`` or the oldest waited ``batch_max_wait_s``.
    batch_idle_s: float = 15.0
    batch_max_wait_s: float = 120.0
    batch_poll_s: float = 30.0
    batch_timeout_s: float = 24 * 3600.0

    def __post_init__(self) -> None:
        if self.mode not in ("online", "batch"):
            raise ValueError(f"gemini mode must be online or batch: {self.mode!r}")


class GeminiClient:
    def __init__(self, settings: GeminiSettings) -> None:
        self.settings = settings
        self._client: Any = None
        self._batch: _BatchDispatcher | None = None
        self._lock = threading.Lock()

    def preflight(self) -> list[str]:
        s = self.settings
        errors = []
        if not s.project:
            errors.append("a Google Cloud project is required (gemini.project)")
        if s.mode == "batch" and not s.batch_bucket.strip().removeprefix("gs://").strip("/"):
            errors.append("batch mode needs a Cloud Storage bucket (gemini.batch_bucket)")
        try:
            from google import genai  # noqa: F401

            if s.mode == "batch":
                from google.cloud import storage  # noqa: F401
        except ImportError as exc:
            errors.append(f"missing package: {exc}")
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
        if media_fps is not None and not 0 < media_fps <= MAX_MEDIA_FPS:
            raise ValueError(f"media_fps must fall in (0, {MAX_MEDIA_FPS}]: {media_fps}")
        media = []
        for raw_path in media_paths or []:
            path = Path(raw_path)
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"media input missing or empty: {path}")
            mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
            media.append(_Media(path=path, mime_type=mime, fps=media_fps if mime.startswith("video/") else None))
        attempts, max_backoff = self.settings.max_retries, 4.0
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if self.settings.mode == "batch":
                    raw_text = self._dispatcher().ask(model, self._request_body(system_instruction, prompt, media), media)
                else:
                    raw_text = self._online(model, system_instruction, prompt, media)
                raw_text = str(raw_text or "").strip()
                if not raw_text:
                    raise RuntimeError("empty response")
                return JsonResponse(raw_text=raw_text, payload=parse_json_response(raw_text), model=model)
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(max_backoff, 0.5 * (2**attempt)))
        raise RuntimeError(f"gemini generation failed after {attempts} attempts: {last_error}")

    def close(self) -> None:
        """Cancel batch jobs that are still running."""
        if self._batch is not None:
            self._batch.close()

    # -- online --------------------------------------------------------
    def _genai(self) -> Any:
        with self._lock:
            if self._client is None:
                from google import genai

                self._client = genai.Client(vertexai=True, project=self.settings.project, location=self.settings.location)
            return self._client

    def _online(self, model: str, system_instruction: str, prompt: str, media: list[_Media]) -> str:
        from google.genai import types

        contents: list[Any] = [prompt]
        for item in media:
            part = types.Part.from_bytes(data=item.path.read_bytes(), mime_type=item.mime_type)
            if item.fps is not None:
                part.video_metadata = types.VideoMetadata(fps=item.fps)
            contents.append(part)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            response_mime_type="application/json",
        )
        return self._genai().models.generate_content(model=model, contents=contents, config=config).text

    # -- batch ---------------------------------------------------------
    def _request_body(self, system_instruction: str, prompt: str, media: list[_Media]) -> dict[str, Any]:
        """One batch request line; each video's ``fileUri`` is filled in once it is uploaded."""
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for item in media:
            part: dict[str, Any] = {"fileData": {"fileUri": None, "mimeType": item.mime_type}}
            if item.fps is not None:
                part["videoMetadata"] = {"fps": item.fps}
            parts.append(part)
        return {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "maxOutputTokens": self.settings.max_output_tokens,
                "responseMimeType": "application/json",
                "temperature": self.settings.temperature,
            },
        }

    def _dispatcher(self) -> _BatchDispatcher:
        with self._lock:
            if self._batch is None:
                self._batch = _BatchDispatcher(self.settings)
            return self._batch


@dataclass(frozen=True)
class _Media:
    path: Path
    mime_type: str
    fps: float | None


@dataclass
class _Pending:
    key: str
    model: str
    body: dict[str, Any]
    media: list[_Media]
    future: Future = field(default_factory=Future)


class _BatchDispatcher:
    """Collects concurrent requests and runs them as Vertex AI batch jobs, one job per window and model."""

    def __init__(self, settings: GeminiSettings) -> None:
        self.settings = settings
        self.bucket_name = settings.batch_bucket.strip().removeprefix("gs://").strip("/")
        self._lock = threading.Lock()
        self._pending: list[_Pending] = []
        self._first_at = self._last_at = 0.0
        self._flusher: threading.Thread | None = None
        self._uploaded: dict[tuple[str, int, int], str] = {}
        self._jobs: list[str] = []
        self._genai: Any = None
        self._bucket: Any = None

    def ask(self, model: str, body: dict[str, Any], media: list[_Media]) -> str:
        item = _Pending(key=uuid.uuid4().hex, model=model, body=body, media=media)
        with self._lock:
            now = time.monotonic()
            if not self._pending:
                self._first_at = now
            self._last_at = now
            self._pending.append(item)
            if self._flusher is None or not self._flusher.is_alive():
                self._flusher = threading.Thread(target=self._flush_loop, name="gemini-batch-flusher", daemon=True)
                self._flusher.start()
        return item.future.result()

    def close(self) -> None:
        with self._lock:
            names = list(self._jobs)
        for name in names:
            try:
                state = _state_name(self._client().batches.get(name=name).state)
                if state not in _TERMINAL:
                    self._client().batches.cancel(name=name)
            except Exception as exc:  # noqa: BLE001 - best effort on shutdown
                LOG.warning("could not cancel batch job %s: %s", name, exc)

    def _client(self) -> Any:
        if self._genai is None:
            from google import genai

            self._genai = genai.Client(vertexai=True, project=self.settings.project, location=self.settings.location)
        return self._genai

    def _storage_bucket(self) -> Any:
        if self._bucket is None:
            from google.cloud import storage

            self._bucket = storage.Client(project=self.settings.project).bucket(self.bucket_name)
        return self._bucket

    def _flush_loop(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._flusher = None
                    return
                now = time.monotonic()
                due = now - self._last_at >= self.settings.batch_idle_s or now - self._first_at >= self.settings.batch_max_wait_s
                batch, self._pending = (self._pending, []) if due else ([], self._pending)
            if not batch:
                time.sleep(min(1.0, self.settings.batch_idle_s))
                continue
            by_model: dict[str, list[_Pending]] = {}
            for item in batch:
                by_model.setdefault(item.model, []).append(item)
            for model, items in by_model.items():
                threading.Thread(target=self._run_job, args=(model, items), name="gemini-batch-job", daemon=True).start()

    def _run_job(self, model: str, items: list[_Pending]) -> None:
        try:
            answers = self._execute(model, items)
        except Exception as exc:  # noqa: BLE001 - handed to every caller of the job
            for item in items:
                item.future.set_exception(exc)
            return
        for item in items:
            answer = answers.get(item.key)
            if isinstance(answer, Exception):
                item.future.set_exception(answer)
            elif answer is None:
                item.future.set_exception(RuntimeError(f"batch output has no line for request {item.key}"))
            else:
                item.future.set_result(answer)

    def _execute(self, model: str, items: list[_Pending]) -> dict[str, Any]:
        from google.genai import types

        bucket = self._storage_bucket()
        distinct = {media.path.resolve(): media for item in items for media in item.media}
        with ThreadPoolExecutor(max_workers=8) as pool:
            uris = dict(zip(distinct, pool.map(lambda media: self._upload_media(bucket, media), distinct.values())))
        lines = []
        for item in items:
            video_parts = [part for part in item.body["contents"][0]["parts"] if "fileData" in part]
            for part, media in zip(video_parts, item.media):
                part["fileData"]["fileUri"] = uris[media.path.resolve()]
            lines.append(json.dumps({"key": item.key, "request": item.body}, ensure_ascii=False))
        job_dir = f"{self.settings.batch_prefix}/jobs/{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        bucket.blob(f"{job_dir}/input.jsonl").upload_from_string("\n".join(lines) + "\n", content_type="application/jsonl")
        job = self._client().batches.create(
            model=model,
            src=f"gs://{self.bucket_name}/{job_dir}/input.jsonl",
            config=types.CreateBatchJobConfig(display_name=f"videobanana-{Path(job_dir).name}", dest=f"gs://{self.bucket_name}/{job_dir}/output/"),
        )
        with self._lock:
            self._jobs.append(job.name)
        LOG.info("batch job %s submitted: model=%s requests=%d", job.name, model, len(items))
        state, error = self._wait(job.name)
        if state not in _SUCCESS:
            raise RuntimeError(f"batch job {job.name} ended in {state}: {error}")
        answers: dict[str, Any] = {}
        for blob in bucket.client.list_blobs(self.bucket_name, prefix=f"{job_dir}/output/"):
            if not blob.name.endswith(".jsonl"):
                continue
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("key") is None:
                    continue
                if record.get("status") or not record.get("response"):
                    answers[record["key"]] = RuntimeError(f"batch request failed in {job.name}: {record.get('status') or 'no response'}")
                else:
                    answers[record["key"]] = types.GenerateContentResponse.model_validate(record["response"]).text
        return answers

    def _upload_media(self, bucket: Any, media: _Media) -> str:
        """Upload a video once (content-addressed) and return its gs:// URI."""
        stat = media.path.stat()
        cache_key = (str(media.path.resolve()), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if cache_key in self._uploaded:
                return self._uploaded[cache_key]
        data = media.path.read_bytes()
        name = f"{self.settings.batch_prefix}/media/{hashlib.sha256(data).hexdigest()}{mimetypes.guess_extension(media.mime_type) or ''}"
        blob = bucket.blob(name)
        if not blob.exists():
            blob.upload_from_string(data, content_type=media.mime_type)
        uri = f"gs://{self.bucket_name}/{name}"
        with self._lock:
            self._uploaded[cache_key] = uri
        return uri

    def _wait(self, name: str) -> tuple[str, Any]:
        started = time.monotonic()
        while True:
            job = self._client().batches.get(name=name)
            state = _state_name(job.state)
            if state in _TERMINAL:
                return state, getattr(job, "error", None)
            if time.monotonic() - started > self.settings.batch_timeout_s:
                self._client().batches.cancel(name=name)
                raise RuntimeError(f"batch job {name} exceeded {self.settings.batch_timeout_s:.0f}s; cancelled")
            time.sleep(self.settings.batch_poll_s)


_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
_SUCCESS = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def _state_name(state: Any) -> str:
    return str(getattr(state, "name", state) or "").split(".")[-1]
