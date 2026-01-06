import tempfile
import subprocess
import os

def audio_to_spectrogram_image(audio_bytes: bytes) -> bytes:
    """
    Bird-tuned spectrogram generator:
    - bandpass focus: 800 Hz to 11 kHz (typical bird vocalization range)
    - log-frequency scale (critical for bird calls)
    - intensity color mapping for high contrast
    - larger resolution for clearer patterns
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        png_path = os.path.join(tmpdir, "spectrogram.png")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Bird-optimized filter:
        # 1) focus band: highpass + lowpass
        # 2) log frequency scale
        # 3) intensity contrast
        # 4) hide legend
        filter_str = (
            "highpass=f=800,"
            "lowpass=f=11000,"
            "showspectrumpic=s=1200x600:scale=log:color=intensity:legend=disabled"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", audio_path,
            "-lavfi", filter_str,
            "-frames:v", "1",
            png_path
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")

        with open(png_path, "rb") as f:
            return f.read()
