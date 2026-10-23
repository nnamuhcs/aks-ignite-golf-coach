"""Bundle the official heavy pose model so inference needs no runtime download."""

import hashlib
import sysconfig
import urllib.request
from pathlib import Path

destination = Path(sysconfig.get_paths()["purelib"]) / (
    "mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite"
)
url = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite"
with urllib.request.urlopen(url, timeout=120) as response:
    model = response.read()
if len(model) < 1_000_000 or model[4:8] != b"TFL3":
    raise RuntimeError("The official pose-model download is not a valid TFLite model")
if hashlib.sha256(model).hexdigest() != "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe":
    raise RuntimeError("The official pose-model download does not match the verified SHA256")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(model)
print(f"Bundled heavy pose model: {len(model)} bytes, SHA256 {hashlib.sha256(model).hexdigest()}")
