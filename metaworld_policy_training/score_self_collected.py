"""
Score self-collected videos using the ReWiND reward model.

Reads existing mp4/gif videos from self_collected_videos/, scores each frame
with the reward model, and generates 1x3 visualisation videos:
  left: original video | middle: raw progress curve | right: diff progress curve

Usage (from metaworld_policy_training/):
    python score_self_collected.py \
        --video_root ../self_collected_videos \
        --reward_model_path ../checkpoints/rewind_metaworld_epoch_19.pth \
        --output_dir score_output/self_collected
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.metaworld import environment_to_instruction
from reward_model.rewind_reward_model import ReWiNDRewardModel
from reward_model.reward_utils import dino_load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dir_name_to_env_id(name):
    """Convert underscore directory name to MetaWorld v2 env id.

    e.g. 'button_press' -> 'button-press-v2'
         'button-press-topdown-v2' -> 'button-press-topdown-v2' (already v2)
    """
    if name.endswith("-v2"):
        return name
    return name.replace("_", "-") + "-v2"


def read_video_frames(video_path):
    """Read video file and return list of RGB numpy arrays (H, W, 3)."""
    frames = iio.imread(video_path)
    # frames shape: (N, H, W, 3) or (N, H, W, 4) for RGBA
    result = []
    for frame in frames:
        if frame.ndim == 2:
            # Grayscale -> RGB
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            # RGBA -> RGB
            frame = frame[:, :, :3]
        result.append(frame.astype(np.uint8))
    return result


def score_trajectory(reward_model, raw_images, text_instruction):
    """Score a trajectory of raw images using ReWiND reward model."""
    text_embedding = reward_model._encode_text_batch([text_instruction])
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32).to(device)

    with torch.inference_mode():
        dino_inputs = [dino_load_image(img) for img in raw_images]
        dino_batches = [
            torch.cat(dino_inputs[i : i + 64])
            for i in range(0, len(dino_inputs), 64)
        ]
        embeddings = []
        for batch in dino_batches:
            emb = reward_model.dino_vits14(batch.to(device)).detach().cpu()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
        video_embeddings = torch.cat(embeddings)

    video_embeddings = reward_model.padding_video(
        video_embeddings, reward_model.args.max_length
    )
    video_embeddings = video_embeddings.unsqueeze(0).float().to(device)

    model = reward_model.rewind_model
    with torch.no_grad():
        progress = model(video_embeddings, text_embedding.float())
        progress = progress.squeeze().cpu().numpy()

    return progress


def save_step_data_csv(csv_path, reward_hat, reward_hat_diff_padded, diff099_padded, diff0999_padded):
    with open(csv_path, "w") as f:
        f.write("step,reward_hat,reward_hat_diff_padded,diff099_padded,diff0999_padded\n")
        for i in range(len(reward_hat)):
            f.write(
                f"{i},{reward_hat[i]:.6f},{reward_hat_diff_padded[i]:.6f},"
                f"{diff099_padded[i]:.6f},{diff0999_padded[i]:.6f}\n"
            )



def generate_video(images, progress_raw, progress_diff, progress_diff_099, progress_diff_0999, video_path, title, fps=10):
    """Generate top-2/bottom-3 MP4: top(video, raw), bottom(0.99 diff, plain diff, 0.999 diff)."""
    from matplotlib.gridspec import GridSpec
    diff_padded = np.concatenate([[0.0], progress_diff])
    diff_099_padded = np.concatenate([[0.0], progress_diff_099])
    diff_0999_padded = np.concatenate([[0.0], progress_diff_0999])
    num_frames = len(images)

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4)

    # Top-left: video frames
    ax_img = fig.add_subplot(gs[0, :3])
    im = ax_img.imshow(images[0])
    ax_img.set_title("Video", fontsize=12)
    ax_img.axis("off")
    step_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Top-right: raw progress
    ax_raw = fig.add_subplot(gs[0, 3:])
    (line_raw,) = ax_raw.plot([], [], "b-", linewidth=2)
    ax_raw.set_xlim(0, num_frames)
    margin = max(0.05, (np.max(progress_raw) - np.min(progress_raw)) * 0.1)
    ax_raw.set_ylim(np.min(progress_raw) - margin, np.max(progress_raw) + margin)
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Progress")
    ax_raw.set_title("Raw Progress", fontsize=12)
    ax_raw.grid(True, alpha=0.3)
    (dot_raw,) = ax_raw.plot([], [], "bo", markersize=5)

    # Bottom-left: 0.99 diff
    ax_d099 = fig.add_subplot(gs[1, :2])
    (line_d099,) = ax_d099.plot([], [], "m-", linewidth=2)
    ax_d099.set_xlim(0, num_frames)
    d099_margin = max(0.01, (np.max(diff_099_padded) - np.min(diff_099_padded)) * 0.15)
    ax_d099.set_ylim(np.min(diff_099_padded) - d099_margin, np.max(diff_099_padded) + d099_margin)
    ax_d099.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_d099.set_xlabel("Step")
    ax_d099.set_ylabel("Diff")
    ax_d099.set_title("100·(0.99·P(s') - P(s))", fontsize=12)
    ax_d099.grid(True, alpha=0.3)
    (dot_d099,) = ax_d099.plot([], [], "mo", markersize=5)

    # Bottom-middle: plain diff
    ax_diff = fig.add_subplot(gs[1, 2:4])
    (line_diff,) = ax_diff.plot([], [], "c-", linewidth=2)
    ax_diff.set_xlim(0, num_frames)
    d_margin = max(0.01, (np.max(diff_padded) - np.min(diff_padded)) * 0.15)
    ax_diff.set_ylim(np.min(diff_padded) - d_margin, np.max(diff_padded) + d_margin)
    ax_diff.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_diff.set_xlabel("Step")
    ax_diff.set_ylabel("Diff")
    ax_diff.set_title("P(s') - P(s)", fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    (dot_diff,) = ax_diff.plot([], [], "co", markersize=5)

    # Bottom-right: 0.999 diff
    ax_d0999 = fig.add_subplot(gs[1, 4:])
    (line_d0999,) = ax_d0999.plot([], [], "g-", linewidth=2)
    ax_d0999.set_xlim(0, num_frames)
    d0999_margin = max(0.01, (np.max(diff_0999_padded) - np.min(diff_0999_padded)) * 0.15)
    ax_d0999.set_ylim(np.min(diff_0999_padded) - d0999_margin, np.max(diff_0999_padded) + d0999_margin)
    ax_d0999.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_d0999.set_xlabel("Step")
    ax_d0999.set_ylabel("Diff")
    ax_d0999.set_title("1000·(0.999·P(s') - P(s))", fontsize=12)
    ax_d0999.grid(True, alpha=0.3)
    (dot_d0999,) = ax_d0999.plot([], [], "go", markersize=5)

    plt.suptitle(title, fontsize=13)

    def init():
        line_raw.set_data([], [])
        line_d099.set_data([], [])
        line_diff.set_data([], [])
        line_d0999.set_data([], [])
        dot_raw.set_data([], [])
        dot_d099.set_data([], [])
        dot_diff.set_data([], [])
        dot_d0999.set_data([], [])
        step_text.set_text("")
        return line_raw, line_d099, line_diff, line_d0999, dot_raw, dot_d099, dot_diff, dot_d0999, step_text, im

    def animate(frame):
        im.set_array(images[frame])
        step_text.set_text(f"Step: {frame}/{num_frames - 1}")
        x = np.arange(frame + 1)
        line_raw.set_data(x, progress_raw[: frame + 1])
        line_d099.set_data(x, diff_099_padded[: frame + 1])
        line_diff.set_data(x, diff_padded[: frame + 1])
        line_d0999.set_data(x, diff_0999_padded[: frame + 1])
        dot_raw.set_data([frame], [progress_raw[frame]])
        dot_d099.set_data([frame], [diff_099_padded[frame]])
        dot_diff.set_data([frame], [diff_padded[frame]])
        dot_d0999.set_data([frame], [diff_0999_padded[frame]])
        return line_raw, line_d099, line_diff, line_d0999, dot_raw, dot_d099, dot_diff, dot_d0999, step_text, im

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(video_path, writer=writer)
    plt.close(fig)


def collect_videos(video_root):
    """Walk video_root and yield (video_path, env_id, category) tuples.

    Handles two directory layouts:
      1. <task_underscore>/<category>/<file>   e.g. button_press/GT/1.mp4
      2. eval_tasks/<env-v2>/<category>/<file>  e.g. eval_tasks/door-close-v2/all_fail/1.mp4
    """
    SKIP_DIRS = {"eval_tasks", "train_tasks"}
    for entry in sorted(os.listdir(video_root)):
        entry_path = os.path.join(video_root, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry in SKIP_DIRS:
            # Second-level dirs are env names in v2 format
            for env_dir in sorted(os.listdir(entry_path)):
                env_dir_path = os.path.join(entry_path, env_dir)
                if not os.path.isdir(env_dir_path):
                    continue
                env_id = dir_name_to_env_id(env_dir)
                for cat in sorted(os.listdir(env_dir_path)):
                    cat_path = os.path.join(env_dir_path, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for vf in sorted(os.listdir(cat_path)):
                        if vf.lower().endswith((".mp4", ".gif")):
                            yield os.path.join(cat_path, vf), env_id, f"{entry}/{env_dir}/{cat}"
        else:
            # Top-level task directories (underscore naming)
            env_id = dir_name_to_env_id(entry)
            for cat in sorted(os.listdir(entry_path)):
                cat_path = os.path.join(entry_path, cat)
                if not os.path.isdir(cat_path):
                    continue
                for vf in sorted(os.listdir(cat_path)):
                    if vf.lower().endswith((".mp4", ".gif")):
                        yield os.path.join(cat_path, vf), env_id, f"{entry}/{cat}"


def main():
    parser = argparse.ArgumentParser(description="Score self-collected videos with ReWiND")
    parser.add_argument("--video_root", type=str, default="../self_collected_videos",
                        help="Root directory containing self-collected videos")
    parser.add_argument("--reward_model_path", type=str, required=True,
                        help="Path to ReWiND checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="score_output/self_collected")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # --- Load ReWiND reward model ---
    print("=== Loading ReWiND Reward Model ===")
    reward_model = ReWiNDRewardModel(
        model_load_path=args.reward_model_path,
        camera_names=["image"],
        device=str(device),
        reward_at_every_step=True,
    )
    max_len = reward_model.args.max_length

    # --- Collect all videos ---
    videos = list(collect_videos(args.video_root))
    print(f"\nFound {len(videos)} videos to score.\n")

    for idx, (video_path, env_id, rel_category) in enumerate(videos):
        # Look up text instruction
        if env_id not in environment_to_instruction:
            print(f"[{idx+1}/{len(videos)}] SKIP {video_path} — unknown env_id '{env_id}'")
            continue

        text_instruction = environment_to_instruction[env_id]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx+1}/{len(videos)}] {rel_category}/{video_name}  env={env_id}  inst=\"{text_instruction}\"")

        # Read frames
        try:
            raw_images = read_video_frames(video_path)
        except Exception as e:
            print(f"  ERROR reading video: {e}")
            continue

        if len(raw_images) < 2:
            print(f"  SKIP — only {len(raw_images)} frame(s)")
            continue

        num_frames = len(raw_images)

        # Score
        progress_subsampled = score_trajectory(reward_model, raw_images, text_instruction)

        # Interpolate back if subsampled
        if num_frames > max_len:
            sampled_indices = np.linspace(0, num_frames - 1, max_len).astype(int)
            progress_raw = np.interp(np.arange(num_frames), sampled_indices, progress_subsampled[:max_len])
        else:
            progress_raw = progress_subsampled[:num_frames]

        progress_diff = progress_raw[1:] - progress_raw[:-1]
        progress_diff_099 = 100 * (0.99 * progress_raw[1:] - progress_raw[:-1])
        progress_diff_0999 = 1000 * (0.999 * progress_raw[1:] - progress_raw[:-1])
        reward_hat_diff_padded = np.concatenate([[0.0], progress_diff])
        diff099_padded = np.concatenate([[0.0], progress_diff_099])
        diff0999_padded = np.concatenate([[0.0], progress_diff_0999])

        # Output path
        out_dir = os.path.join(args.output_dir, rel_category)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{video_name}_scored.mp4")
        csv_path = os.path.join(out_dir, f"{video_name}_step_data.csv")

        save_step_data_csv(csv_path, progress_raw, reward_hat_diff_padded, diff099_padded, diff0999_padded)

        title = f"{rel_category}/{video_name} — {env_id}"
        generate_video(raw_images, progress_raw, progress_diff, progress_diff_099, progress_diff_0999, out_path, title, fps=args.fps)
        print(f"  -> {out_path}  ({num_frames} frames, progress [{progress_raw.min():.3f}, {progress_raw.max():.3f}])")
        print(f"  -> {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
