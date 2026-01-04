import os
import io
from PIL import Image
import ffmpeg
import tempfile

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
AUDIO_TRIM_SECONDS = int(os.getenv("AUDIO_TRIM_SECONDS", "6"))

def resize_image(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()

def trim_audio(audio_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in_audio")
        out_path = os.path.join(tmpdir, "out.wav")

        with open(in_path, "wb") as f:
            f.write(audio_bytes)

        (
            ffmpeg
            .input(in_path)
            .output(out_path, t=AUDIO_TRIM_SECONDS, ac=1, ar=22050)
            .overwrite_output()
            .run(quiet=True)
        )

        with open(out_path, "rb") as f:
            return f.read()
