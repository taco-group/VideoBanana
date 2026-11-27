"""Turn execution with CoinVE-Edit: edit the entry video, verify each attempt, report the candidates.

For each attempt of a turn:

1. The intent compiler checks the instructions (``conversation/compiler.py``);
   if it blocks them as ambiguous or gives no valid answer, the attempt makes
   no edit.
2. The entry video is prepared for the editor: a video longer than 49 frames
   is cut to its first 49 frames (lossless H.264); anything else is passed as is.
3. The editor receives the instruction writer's instructions, each once and in
   order, in one call with seed ``seed_base + attempt - 1`` (0, then 1). The
   writer returns at most ``max_instructions_per_call`` instructions, so one
   call carries them all and every output is one generation from its entry.
4. The verifier scores the output against the video the editor received.

Attempts stop at the first candidate that passes the gate. Instructions, editor
calls and verdicts are saved under ``turns/Tnn/``, so an interrupted turn
resumes without repeating finished work.

CoinVE-Edit runs in its own environment as persistent workers, one per GPU,
each loading the model once (``coinve_worker.py``).
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .conversation.compiler import IntentCompiler
from .conversation.contracts import CandidateReport, TurnResult, TurnSpec
from .conversation.instructions import EditorInstruction, InstructionWriter
from .conversation.scoring import passes_gate
from .conversation.verifier import MultiTurnVerifier

#: Longest input CoinVE-Edit takes.
MAX_FRAMES = 49
VERDICT_FILE = "multiturn_verdict.json"
EDIT_FILE = "edit.json"
COMPILE_FILE = "intent_compiler.json"
WORKER_SCRIPT = Path(__file__).with_name("coinve_worker.py")


class EditorError(RuntimeError):
    """The editor call failed (worker crash, out of memory, missing file, ...)."""


@dataclass(frozen=True)
class EditRequest:
    """One editor call."""

    input_video: str
    prompts: list[str]
    seed: int
    #: Frames the editor reads from ``input_video``.
    frames: int
    output_path: str


class Editor(Protocol):
    def edit(self, request: EditRequest) -> None:
        """Write the edited video to ``request.output_path``; raise :class:`EditorError` on failure."""


# -- input preparation --------------------------------------------------------


def frame_count(path: str | Path) -> int:
    """Frame count from the container header; raises if the video cannot be decoded."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok or count <= 0:
        raise ValueError(f"video is not decodable: {path}")
    return count


def prepare_input(video: str, destination: Path, max_frames: int = MAX_FRAMES) -> tuple[str, int]:
    """The video the editor receives and its frame count.

    A video within ``max_frames`` is used as is. A longer one is cut to its
    first ``max_frames`` frames, losslessly (H.264, qp 0), so they decode exactly
    as the editor would read them from the video itself.
    """
    count = frame_count(video)
    if count <= max_frames:
        return video, count
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
         "-frames:v", str(max_frames), "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-an", str(destination)],
        check=True,
    )
    if frame_count(destination) != max_frames:
        raise RuntimeError(f"prepared editor input does not have {max_frames} frames: {destination}")
    return str(destination), max_frames


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("cutting a long video needs ffmpeg (imageio-ffmpeg or on PATH)")
        return found
    return imageio_ffmpeg.get_ffmpeg_exe()


def editor_prompts(instructions: list[EditorInstruction]) -> list[str]:
    """The instructions as the editor receives them: each text once, in order, stripped."""
    return [text.strip() for text in dict.fromkeys(item.text for item in instructions)]


# -- turn executor ------------------------------------------------------------


class EditorTurnExecutor:
    """Runs a turn's attempts from the entry video; see the module docstring."""

    def __init__(
        self,
        root: str | Path,
        *,
        writer: InstructionWriter,
        verifier: MultiTurnVerifier,
        editor: Editor,
        compiler: IntentCompiler | None = None,
        gate_floor: int = 2,
        item_gate_floor: int | None = None,
        max_frames: int = MAX_FRAMES,
        seed_base: int = 0,
        editor_tries: int = 3,
    ) -> None:
        self.root = Path(root)
        self.writer = writer
        self.verifier = verifier
        self.editor = editor
        self.compiler = compiler
        #: The same floors as the conversation runner: resampling stops at the first passing candidate.
        self.gate_floor = gate_floor
        self.item_gate_floor = item_gate_floor
        self.max_frames = max_frames
        self.seed_base = seed_base
        #: Calls per attempt when the editor fails; the retries use the same seed.
        self.editor_tries = editor_tries

    def run_turn(self, spec: TurnSpec) -> TurnResult:
        turn_root = self.root / f"T{spec.turn:02d}"
        self._claim(spec, turn_root)
        prompts = editor_prompts(self._instructions(spec, turn_root))
        candidates: list[CandidateReport] = []
        for attempt in range(1, spec.max_attempts + 1):
            attempt_root = turn_root / f"A{attempt}"
            candidate_id = f"T{spec.turn:02d}_A{attempt}"
            if not self._compiled(prompts, attempt_root, tag=f"T{spec.turn:02d}_A{attempt}"):
                candidates.append(CandidateReport(candidate_id=candidate_id, video_path="", structural_passed=False))
                continue
            edited = self._edit(spec.entry.video_path, prompts, attempt_root, seed=self.seed_base + attempt - 1)
            if edited is None:
                candidates.append(CandidateReport(candidate_id=candidate_id, video_path="", structural_passed=False))
                continue
            # Judge against the frames the editor received, so both videos share one timeline.
            editor_input, output = edited
            report = self._verdict(spec, attempt_root, candidate_id, editor_input, output)
            candidates.append(report)
            if passes_gate(report, spec, self.gate_floor, self.item_gate_floor):
                break
        return TurnResult(candidates=candidates, sub_session_id=str(turn_root))

    @staticmethod
    def spec_fingerprint(spec: TurnSpec) -> dict[str, object]:
        """What a turn's saved instructions and videos were made for."""
        return {
            "entry": spec.entry.node_id,
            "entry_video": spec.entry.video_path,
            **{
                role: [[item.intent_id, item.text] for item in getattr(spec, role)]
                for role in ("execute", "preserve", "observe", "overwritten")
            },
        }

    def _claim(self, spec: TurnSpec, turn_root: Path) -> None:
        """Reuse a turn directory only for the spec it was made for; move any other aside."""
        path = turn_root / "turn_spec.json"
        wanted = self.spec_fingerprint(spec)
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != wanted:
            turn_root.rename(turn_root.with_name(f"{turn_root.name}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"))
        if not path.is_file():
            turn_root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(wanted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _instructions(self, spec: TurnSpec, turn_root: Path) -> list[EditorInstruction]:
        """Write the turn's instructions once; a resumed turn reuses them."""
        path = turn_root / "instructions.json"
        if path.is_file():
            return [EditorInstruction.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
        instructions = self.writer.write(spec)
        path.write_text(json.dumps([item.model_dump() for item in instructions], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return instructions

    def _compiled(self, prompts: list[str], attempt_root: Path, *, tag: str) -> bool:
        """Whether the intent compiler accepts the instructions; its answer is saved for a resumed turn."""
        if self.compiler is None:
            return True
        path = attempt_root / COMPILE_FILE
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("instructions") == prompts:
                return saved["status"] == "ok"
        attempt_root.mkdir(parents=True, exist_ok=True)
        result = self.compiler.compile(prompts, tag=tag)
        record = {"instructions": prompts, "status": result.status, "reason": result.reason, "intents": result.intents}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result.ok

    def _edit(self, entry_video: str, prompts: list[str], attempt_root: Path, *, seed: int) -> tuple[str, str] | None:
        """(editor input, output video) of one attempt, or ``None`` when the editor produced no usable video."""
        attempt_root.mkdir(parents=True, exist_ok=True)
        output = (attempt_root / "output.mp4").resolve()
        editor_input, frames = prepare_input(entry_video, attempt_root / "normalized_input.mp4", self.max_frames)
        request = EditRequest(input_video=editor_input, prompts=prompts, seed=seed, frames=frames, output_path=str(output))
        record_path = attempt_root / EDIT_FILE
        if record_path.is_file():
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            if saved["request"] == asdict(request):
                if saved["status"] == "succeeded" and output.is_file():
                    return editor_input, str(output)
                if saved["status"] == "failed":
                    return None
        error = _run_editor(self.editor, request, self.editor_tries)
        record = {"request": asdict(request), "status": "failed" if error else "succeeded", "error": error}
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return None if error else (editor_input, str(output))

    def _verdict(self, spec: TurnSpec, attempt_root: Path, candidate_id: str, editor_input: str, output: str) -> CandidateReport:
        """The saved verdict of this attempt if the same verifier made it, else a new one."""
        path = attempt_root / VERDICT_FILE
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.pop("verifier_identity", None) == self.verifier.identity():
                saved.pop("verifier", None)
                report = CandidateReport.model_validate(saved)
                if report.candidate_id == candidate_id and report.video_path == output:
                    return report
        scores = self.verifier.score_pair(spec, editor_input, output, tag=f"T{spec.turn:02d}_{attempt_root.name}")
        report = CandidateReport(
            candidate_id=candidate_id,
            video_path=output,
            generations=1,
            structural_passed=True,
            instruction_scores=scores.instruction_scores,
            retention_scores=scores.retention_scores,
            nontarget_score=scores.nontarget_score,
            quality_score=scores.quality_score,
        )
        payload = {"verifier": self.verifier.name, "verifier_identity": self.verifier.identity(), **report.model_dump(mode="json")}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return report


def _run_editor(editor: Editor, request: EditRequest, tries: int) -> str | None:
    """Call the editor and check its output; the error message, or ``None`` on success."""
    if not request.prompts or not all(request.prompts):
        return "empty editor instruction"
    error = ""
    for _ in range(tries):
        try:
            editor.edit(request)
            break
        except EditorError as exc:
            error = str(exc)
    else:
        return error or "editor failed"
    try:
        frames = frame_count(request.output_path)
    except ValueError as exc:
        return str(exc)
    if frames > request.frames:
        return f"editor output has {frames} frames, more than its input's {request.frames}"
    return None


# -- CoinVE-Edit --------------------------------------------------------------


@dataclass(frozen=True)
class CoinVESettings:
    """Where CoinVE-Edit and its weights are, and how it is run."""

    repo_root: str
    python: str
    composite_checkpoint: str
    local_model_path: str
    mllm_model: str
    #: Default: ``<repo_root>/infer_coinve_single.py``.
    entrypoint: str = ""
    gpus: tuple[int, ...] = (0,)
    #: 1280 x 720.
    eval_max_pixels: int = 921600
    num_inference_steps: int = 50
    #: Seconds to wait for one edit before the worker is killed and restarted.
    job_timeout: float = 3600.0

    @property
    def entrypoint_path(self) -> str:
        return self.entrypoint or str(Path(self.repo_root) / "infer_coinve_single.py")


def coinve_argv(settings: CoinVESettings, request: EditRequest) -> list[str]:
    """Arguments of ``infer_coinve_single.py`` for one call."""
    output = Path(request.output_path).resolve()
    return [
        "--src_video", str(Path(request.input_video).resolve()),
        "--prompts", *request.prompts,
        "--composite_checkpoint", settings.composite_checkpoint,
        "--local_model_path", settings.local_model_path,
        "--mllm_model", settings.mllm_model,
        "--output_dir", str(output.parent),
        "--output_name", output.stem,
        "--eval_max_pixels", str(int(settings.eval_max_pixels)),
        "--eval_max_frame", str(request.frames),
        "--src_max_frames", str(request.frames),
        "--seed", str(request.seed),
        "--num_inference_steps", str(settings.num_inference_steps),
        "--no_side_by_side",
        "--save_instruction_masks",
    ]


class CoinVEEditor:
    """CoinVE-Edit behind one persistent worker per GPU; a call takes whichever worker is free."""

    def __init__(self, settings: CoinVESettings, log_dir: str | Path) -> None:
        self.settings = settings
        self._idle: queue.Queue[_Worker] = queue.Queue()
        for gpu in settings.gpus:
            self._idle.put(_Worker(settings, gpu, Path(log_dir) / f"coinve_worker_gpu{gpu}.log"))
        self._workers = list(self._idle.queue)

    def preflight(self) -> list[str]:
        s = self.settings
        required = {
            "CoinVE-Edit repository": s.repo_root,
            "CoinVE-Edit Python": s.python,
            "CoinVE-Edit entrypoint": s.entrypoint_path,
            "composite checkpoint": s.composite_checkpoint,
            "local model root": s.local_model_path,
            "MLLM model": s.mllm_model,
        }
        return [f"{label} missing: {path}" for label, path in required.items() if not Path(path).exists()]

    def edit(self, request: EditRequest) -> None:
        worker = self._idle.get()
        try:
            worker.run(coinve_argv(self.settings, request))
        finally:
            self._idle.put(worker)

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()


class _Worker:
    """One ``coinve_worker.py`` process on one GPU, started on first use and after a crash."""

    def __init__(self, settings: CoinVESettings, gpu: int, log_path: Path) -> None:
        self.settings = settings
        self.gpu = gpu
        self.log_path = log_path
        self._process: subprocess.Popen[str] | None = None
        self._log: Any = None
        self._lock = threading.Lock()

    def run(self, argv: list[str]) -> dict[str, Any]:
        with self._lock:
            process = self._ensure()
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps({"argv": argv}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = self._read_answer(process)
            except OSError as exc:
                line, error = "", str(exc)
            except TimeoutError:
                process.kill()  # a hung worker; the next call starts a new one
                self._stop_locked()
                raise EditorError(
                    f"CoinVE-Edit worker on GPU {self.gpu} gave no answer in {self.settings.job_timeout:.0f} s; "
                    f"killed, see {self.log_path}"
                ) from None
            else:
                error = ""
            if not line:
                self._stop_locked()
                raise EditorError(f"CoinVE-Edit worker on GPU {self.gpu} exited {error}; see {self.log_path}")
            result = json.loads(line)
            if result.get("status") != "success":
                raise EditorError(str(result.get("error") or "CoinVE-Edit job failed"))
            return result

    def _read_answer(self, process: subprocess.Popen[str]) -> str:
        """The worker's answer line; ``TimeoutError`` after ``job_timeout`` seconds."""
        answers: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: answers.put(process.stdout.readline()), daemon=True).start()
        try:
            return answers.get(timeout=self.settings.job_timeout)
        except queue.Empty:
            raise TimeoutError from None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _ensure(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._stop_locked()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu), "PYTHONUNBUFFERED": "1"}
        self._process = subprocess.Popen(
            [self.settings.python, str(WORKER_SCRIPT), "--entrypoint", self.settings.entrypoint_path],
            cwd=self.settings.repo_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self._process

    def _stop_locked(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.close()  # end of input: the worker exits
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001 - a stuck worker is killed
                process.kill()
                process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None
