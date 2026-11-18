import json
import os
import subprocess
import tempfile


def write_cuts(cuts, path, source):
    with open(path, "w") as f:
        json.dump({"source": source,
                   "segments": [{"start": s, "end": e, "confidence": c} for s, e, c in cuts]},
                  f, indent=2)


def read_cuts(path):
    with open(path) as f:
        data = json.load(f)
    return [(seg["start"], seg["end"], seg.get("confidence", 1.0)) for seg in data["segments"]]


def render(source, cuts, output_path, crf=18):
    """Cut each segment from the ORIGINAL file (full resolution, audio kept), re-encoding
    so cuts are frame-accurate, then join them losslessly with the concat demuxer."""
    if not cuts:
        raise ValueError("No segments to render")
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (start, end, _) in enumerate(cuts):
            part = os.path.join(tmp, f"part_{i:04d}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                            "-i", source, "-map", "0:v:0", "-map", "0:a:0?",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", part], check=True)
            parts.append(part)
        list_path = os.path.join(tmp, "parts.txt")
        with open(list_path, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                        "-c", "copy", "-movflags", "+faststart", output_path], check=True)
