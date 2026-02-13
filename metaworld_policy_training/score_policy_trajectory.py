"""
Score a trained policy's trajectory using the ReWiND reward model.

Generates:
  1. correlation_analysis.txt  – Pearson/Spearman between reward model outputs and GT reward
  2. trajectory_analysis.mp4   – 2×2 video (env | raw progress | diff progress | GT reward)

Usage (run from metaworld_policy_training/):
    python score_expert_trajectory.py \
        --env_id button-press-wall-v2 \
        --best_model_path logs/<group_name>/best_model.zip \
        --reward_model_path ../checkpoints/rewind_metaworld_epoch_19.pth \
        --output_dir score_output
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.metaworld import (
    MetaworldBase,
    MetaworldImageEmbeddingWrapper,
    environment_to_instruction,
    create_wrapped_env,
)
from reward_model.rewind_reward_model import ReWiNDRewardModel
from reward_model.policy_observation_encoder import PolicyObservationEncoder
from reward_model.reward_utils import dino_load_image

from offline_rl_algorithms.rlpd import RLPD
from offline_rl_algorithms.iql import IQL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(best_model_path, env):
    """Load best_model.zip (SB3 format). Try RLPD first, then IQL."""
    errors = {}
    for cls in [RLPD, IQL]:
        try:
            # load() is an instance method that uses self.__class__ to construct a new model
            dummy = cls.__new__(cls)
            model = dummy.load(
                best_model_path,
                env=env,
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )
            print(f"Loaded policy as {cls.__name__} from {best_model_path}")
            return model
        except Exception as e:
            errors[cls.__name__] = e
            continue
    for name, e in errors.items():
        print(f"  {name} load error: {e}")
    raise RuntimeError(f"Failed to load {best_model_path} with RLPD or IQL")


def run_episode(env, policy, max_steps=128):
    """Roll out one episode, collecting raw images, GT rewards, and success info."""
    obs = env.reset()
    raw_images = []
    gt_rewards = []
    success = False
    success_step = None

    episode_start = np.array([True])
    for step in range(max_steps):
        # Collect the raw RGB image before stepping
        raw_img = env.envs[0].render(mode="rgb_array")
        raw_images.append(raw_img)

        action, _ = policy.predict(obs, deterministic=True, episode_start=episode_start)
        episode_start = np.array([False])
        obs, reward, done, info = env.step(action)
        gt_rewards.append(reward[0])

        if info[0].get("success", False) and not success:
            success = True
            success_step = step

        if done[0]:
            # Render the post-final frame before exiting
            raw_img = env.envs[0].render(mode="rgb_array")
            raw_images.append(raw_img)
            gt_rewards.append(reward[0])
            break

    return raw_images, np.array(gt_rewards), success, success_step


def score_trajectory(reward_model, raw_images, text_instruction):
    """Score a trajectory of raw images using ReWiND reward model.
    Returns per-frame progress values (0~1).
    """
    # Encode text
    text_embedding = reward_model._encode_text_batch([text_instruction])
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32).to(device)

    # Encode all frames with DINO
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
        video_embeddings = torch.cat(embeddings)  # (num_frames, 768)

    # Always pad/subsample to max_length (attention mask is fixed size)
    video_embeddings = reward_model.padding_video(
        video_embeddings, reward_model.args.max_length
    )

    video_embeddings = video_embeddings.unsqueeze(0).float().to(device)  # (1, T, 768)

    # Forward pass
    model = reward_model.rewind_model
    with torch.no_grad():
        progress = model(video_embeddings, text_embedding.float())  # (1, T, 1)
        progress = progress.squeeze().cpu().numpy()  # (T,)

    return progress


def compute_correlations(x, y, label_x, label_y):
    """Compute Pearson and Spearman correlations, return dict."""
    if len(x) < 3:
        return {"pearson": float("nan"), "p_pearson": float("nan"),
                "spearman": float("nan"), "label": f"{label_x} vs {label_y}", "n": len(x)}
    p_r, p_p = pearsonr(x, y)
    s_r, _ = spearmanr(x, y)
    return {"pearson": p_r, "p_pearson": p_p, "spearman": s_r,
            "label": f"{label_x} vs {label_y}", "n": len(x)}


def write_correlation_report(path, env_id, results, success, success_step, num_frames):
    """Write correlation analysis to text file."""
    with open(path, "w") as f:
        f.write(f"Correlation Analysis for {env_id}\n")
        f.write(f"Episode: {num_frames} frames, success={success}, success_step={success_step}\n\n")
        for r in results:
            f.write(f"--- {r['label']} (n={r['n']}) ---\n")
            f.write(f"  Pearson:  {r['pearson']:.6f} (p={r['p_pearson']:.2e})\n")
            f.write(f"  Spearman: {r['spearman']:.6f}\n\n")
    print(f"Saved correlation analysis to {path}")


def generate_video(images, progress_raw, progress_diff, gt_rewards,
                   video_path, env_id, success_step, fps=20):
    """Generate 2x2 MP4: top-left=env, top-right=raw progress, bottom-left=diff, bottom-right=GT reward."""
    # For diff, we have one fewer frame; pad with 0 at the start for alignment
    diff_padded = np.concatenate([[0.0], progress_diff])
    num_frames = len(images)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top-left: env video ---
    ax_img = axes[0, 0]
    im = ax_img.imshow(images[0])
    ax_img.set_title("Environment", fontsize=12)
    ax_img.axis("off")
    step_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # --- Top-right: raw progress ---
    ax_raw = axes[0, 1]
    (line_raw,) = ax_raw.plot([], [], "b-", linewidth=2)
    ax_raw.set_xlim(0, num_frames)
    margin = max(0.05, (np.max(progress_raw) - np.min(progress_raw)) * 0.1)
    ax_raw.set_ylim(np.min(progress_raw) - margin, np.max(progress_raw) + margin)
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Progress")
    ax_raw.set_title("ReWiND Raw Progress", fontsize=12)
    ax_raw.grid(True, alpha=0.3)
    (dot_raw,) = ax_raw.plot([], [], "bo", markersize=5)

    # --- Bottom-left: diff progress ---
    ax_diff = axes[1, 0]
    (line_diff,) = ax_diff.plot([], [], "m-", linewidth=2)
    ax_diff.set_xlim(0, num_frames)
    d_margin = max(0.01, (np.max(diff_padded) - np.min(diff_padded)) * 0.15)
    ax_diff.set_ylim(np.min(diff_padded) - d_margin, np.max(diff_padded) + d_margin)
    ax_diff.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_diff.set_xlabel("Step")
    ax_diff.set_ylabel("Diff")
    ax_diff.set_title("ReWiND Diff (P(s')-P(s))", fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    (dot_diff,) = ax_diff.plot([], [], "mo", markersize=5)

    # --- Bottom-right: GT reward ---
    ax_gt = axes[1, 1]
    (line_gt,) = ax_gt.plot([], [], "r-", linewidth=2)
    ax_gt.set_xlim(0, num_frames)
    g_margin = max(0.1, (np.max(gt_rewards) - np.min(gt_rewards)) * 0.1)
    ax_gt.set_ylim(np.min(gt_rewards) - g_margin, np.max(gt_rewards) + g_margin)
    ax_gt.set_xlabel("Step")
    ax_gt.set_ylabel("GT Reward")
    ax_gt.set_title("GT Reward", fontsize=12)
    ax_gt.grid(True, alpha=0.3)
    (dot_gt,) = ax_gt.plot([], [], "ro", markersize=5)

    plt.suptitle(f"Trajectory Analysis - {env_id}", fontsize=14)
    plt.tight_layout()

    def init():
        for ln in [line_raw, line_diff, line_gt]:
            ln.set_data([], [])
        for d in [dot_raw, dot_diff, dot_gt]:
            d.set_data([], [])
        step_text.set_text("")
        return line_raw, line_diff, line_gt, dot_raw, dot_diff, dot_gt, step_text, im

    def animate(frame):
        im.set_array(images[frame])
        status = " SUCCESS!" if success_step is not None and frame >= success_step else ""
        step_text.set_text(f"Step: {frame}/{num_frames - 1}{status}")
        x = np.arange(frame + 1)
        line_raw.set_data(x, progress_raw[: frame + 1])
        line_diff.set_data(x, diff_padded[: frame + 1])
        line_gt.set_data(x, gt_rewards[: frame + 1])
        dot_raw.set_data([frame], [progress_raw[frame]])
        dot_diff.set_data([frame], [diff_padded[frame]])
        dot_gt.set_data([frame], [gt_rewards[frame]])
        return line_raw, line_diff, line_gt, dot_raw, dot_diff, dot_gt, step_text, im

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    print(f"Saving video to {video_path} ({num_frames} frames)...")
    anim.save(video_path, writer=writer)
    plt.close(fig)
    print(f"Saved video to {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Score trained policy trajectory with ReWiND")
    parser.add_argument("--env_id", type=str, default="button-press-wall-v2")
    parser.add_argument("--best_model_path", type=str, required=True,
                        help="Path to best_model.zip from training logs")
    parser.add_argument("--reward_model_path", type=str, required=True,
                        help="Path to ReWiND checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="score_output")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    text_instruction = environment_to_instruction[args.env_id]
    print(f"Environment: {args.env_id}")
    print(f"Instruction: {text_instruction}")

    # --- Load ReWiND reward model ---
    print("\n=== Loading ReWiND Reward Model ===")
    reward_model = ReWiNDRewardModel(
        model_load_path=args.reward_model_path,
        camera_names=["image"],
        device=str(device),
        reward_at_every_step=True,
    )

    # --- Create wrapped environment (same as training) ---
    print("\n=== Creating Environment ===")
    lang_feat = reward_model.encode_text_for_policy(text_instruction).squeeze()
    env_fn = create_wrapped_env(
        args.env_id,
        reward_model=reward_model,
        language_features=lang_feat,
        text_instruction=text_instruction,
        success_bonus=0.0,  # we want raw GT reward
        monitor=True,
        goal_observable=True,
        is_state_based=False,
        mode="eval",
        use_proprio=True,
        action_chunk_size=10,
        terminate_on_success=True,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([env_fn])

    # --- Load trained policy ---
    print("\n=== Loading Trained Policy ===")
    policy = load_policy(args.best_model_path, env)

    # --- Run episode ---
    print("\n=== Running Episode ===")
    raw_images, gt_rewards, success, success_step = run_episode(
        env, policy, max_steps=args.max_steps
    )
    num_frames = len(raw_images)
    print(f"Episode: {num_frames} steps, success={success}, success_step={success_step}")

    # --- Score trajectory with ReWiND ---
    print("\n=== Scoring Trajectory ===")
    progress_subsampled = score_trajectory(reward_model, raw_images, text_instruction)

    # Reward model subsamples to max_length frames; interpolate back to full trajectory
    max_len = reward_model.args.max_length
    if num_frames > max_len:
        sampled_indices = np.linspace(0, num_frames - 1, max_len).astype(int)
        progress_raw = np.interp(np.arange(num_frames), sampled_indices, progress_subsampled[:max_len])
    else:
        progress_raw = progress_subsampled[:num_frames]

    print(f"  progress_raw length: {len(progress_raw)}, gt_rewards length: {len(gt_rewards)}")

    # Compute diff
    progress_diff = np.diff(progress_raw)  # length = num_frames - 1

    # --- Correlation analysis ---
    print("\n=== Computing Correlations ===")
    results = []

    # Raw progress vs GT reward (all steps)
    results.append(compute_correlations(
        progress_raw, gt_rewards, "Raw Progress", "GT Reward"))

    # Diff progress vs GT reward (aligned: diff[i] corresponds to gt[i+1])
    results.append(compute_correlations(
        progress_diff, gt_rewards[1:], "Diff Progress", "GT Reward"))

    # First 30 steps
    n30 = min(30, num_frames)
    results.append(compute_correlations(
        progress_raw[:n30], gt_rewards[:n30], "Raw Progress (first 30)", "GT Reward (first 30)"))

    n30d = min(30, len(progress_diff))
    results.append(compute_correlations(
        progress_diff[:n30d], gt_rewards[1 : n30d + 1], "Diff Progress (first 30)", "GT Reward (first 30)"))

    # Pre-success
    if success_step is not None and success_step > 2:
        results.append(compute_correlations(
            progress_raw[: success_step + 1], gt_rewards[: success_step + 1],
            "Raw Progress (pre-success)", "GT Reward (pre-success)"))
        results.append(compute_correlations(
            progress_diff[:success_step], gt_rewards[1 : success_step + 1],
            "Diff Progress (pre-success)", "GT Reward (pre-success)"))

    for r in results:
        print(f"  {r['label']} (n={r['n']}): Pearson={r['pearson']:.4f}, Spearman={r['spearman']:.4f}")

    corr_path = os.path.join(args.output_dir, "correlation_analysis.txt")
    write_correlation_report(corr_path, args.env_id, results, success, success_step, num_frames)

    # --- Generate video ---
    print("\n=== Generating Video ===")
    video_path = os.path.join(args.output_dir, "trajectory_analysis.mp4")
    generate_video(raw_images, progress_raw, progress_diff, gt_rewards,
                   video_path, args.env_id, success_step, fps=args.fps)

    print("\nDone!")


if __name__ == "__main__":
    main()
