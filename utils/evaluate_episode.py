"""
Simplified episode loader + video writer.
"""

import argparse
import pathlib
import numpy as np
import imageio
from PIL import Image

# helpers
def _normalize_loaded_to_frames(loaded: np.ndarray):
    """
    Convert a 4D numpy array (T,H,W,C) into a list of HxWxC uint8 frames.
    """
    if not isinstance(loaded, np.ndarray) or loaded.ndim != 4:
        raise ValueError(f"Expected 4D ndarray (T,H,W,C), got {type(loaded)} with shape {getattr(loaded, 'shape', None)}")

    frames = []
    for f in loaded:
        f = np.asarray(f)
        if f.dtype != np.uint8:
            if np.issubdtype(f.dtype, np.floating):
                if f.max() <= 1.0:
                    f = (f * 255).round().astype(np.uint8)
                else:
                    f = f.round().astype(np.uint8)
            else:
                f = np.clip(f, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames

# core loader
def load_episode_pixels(episode_path: pathlib.Path):
    """
    Load an episode file (.npy, .npz, or .pth) and return a list of HxWxC uint8 frames.
    """
    episode_path = pathlib.Path(episode_path)
    if not episode_path.exists():
        raise FileNotFoundError(f"{episode_path} does not exist")

    if episode_path.suffix == '.npz':
        npz = np.load(episode_path, allow_pickle=False)
        loaded = npz[list(npz.files)[0]]
    elif episode_path.suffix == '.npy':
        loaded = np.load(episode_path)
    elif episode_path.suffix in ('.pth', '.pt'):
        import torch
        loaded = torch.load(episode_path, map_location='cpu', weights_only=False)
        if hasattr(loaded, 'numpy'):
            loaded = loaded.numpy()
    else:
        raise ValueError(f"Unsupported file extension: {episode_path.suffix}")

    frames = _normalize_loaded_to_frames(loaded)
    return frames

# video writer
def write_video(frames, out_path, fps=30):
    if len(frames) == 0:
        raise ValueError("No frames to write")

    H, W, C = frames[0].shape
    # Resize frames if needed
    for i, f in enumerate(frames):
        if f.shape != (H, W, C):
            f = Image.fromarray(f).resize((W, H), Image.BILINEAR)
            frames[i] = np.asarray(f)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264',
                                quality=8, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", required=True,
                        help="Path to save folder that contains 'obses' subfolder")
    parser.add_argument("--episode", type=int, required=True,
                        help="Episode index (integer)")
    parser.add_argument("--out", required=True,
                        help="Output video path (e.g. ep00000.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()
    args.episode = args.episode - 1 # fix indexing
    save_folder = pathlib.Path(args.save_folder)
    obses_folder = save_folder / "obses"
    candidates = [
        obses_folder / f"episode_{args.episode:05d}.pth",
        obses_folder / f"episode_{args.episode:05d}.pt",
        obses_folder / f"episode_{args.episode:05d}.npz",
        obses_folder / f"episode_{args.episode:05d}.npy",
    ]

    epfile = next((c for c in candidates if c.exists()), None)
    if epfile is None:
        raise FileNotFoundError(f"No episode file found for index {args.episode} in {obses_folder}")

    print(f"Loading episode from {epfile} ...")
    frames = load_episode_pixels(epfile)
    print(f"Loaded {len(frames)} frames. Writing video to {args.out} ...")
    write_video(frames, args.out, fps=args.fps)
    print("Done.")

if __name__ == "__main__":
    main()
