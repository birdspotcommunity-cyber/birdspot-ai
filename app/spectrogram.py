import tempfile
import subprocess
import os

def audio_to_spectrogram_image(audio_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        png_path = os.path.join(tmpdir, "spectrogram.png")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", audio_path,
            "-lavfi", "showspectrumpic=s=1024x512:mode=separate:color=channel",
            "-frames:v", "1",
            png_path
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")

        with open(png_path, "rb") as f:
            return f.read()
