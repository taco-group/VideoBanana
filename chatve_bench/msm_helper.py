"""MSM (motion smoothness) of a list of videos, run with the CoVEBench Python environment.

Called by ``chatve_bench.vq``: ``python msm_helper.py <covebench dir> <videos.json>`` prints {video: score} as JSON.
A video that cannot be scored is left out and reported on stderr.
Ten evenly spaced frames, grayscale, resized to 224x224, scored with CoVEBench's ``motion_smoothness``.
"""

import json
import math
import sys
from pathlib import Path

covebench = Path(sys.argv[1])
sys.path[:0] = [str(covebench / "src"), str(covebench / "metrics" / "vq" / "msm")]

import cv2  # noqa: E402
from covebench_eval.common import equal_indices  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from eval_msm import motion_smoothness  # noqa: E402


def frames(path: str, count: int = 10, size: tuple[int, int] = (224, 224)):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    batch = vr.get_batch(equal_indices(len(vr), count)).asnumpy()
    return [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA) for f in batch]


scores = {}
for v in json.loads(Path(sys.argv[2]).read_text()):
    try:
        s = motion_smoothness(frames(v))
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the others
        print(f"MSM failed for {v}: {e}", file=sys.stderr)
        continue
    if math.isnan(s):
        print(f"MSM undefined for {v} (fewer than 3 frames)", file=sys.stderr)
        continue
    scores[v] = s
print(json.dumps(scores))
