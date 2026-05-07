import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def padding_video(video_frames, max_length):
    video_length = len(video_frames)
    if type(video_frames) == np.ndarray:
        video_frames = torch.tensor(video_frames)
    if video_length < max_length:
        # padding first frame
        padding_length = max_length - video_length
        first_frame = video_frames[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_length, 1)
        video_frames = torch.cat([padding_frames, video_frames], dim=0)
    
    elif video_length > max_length:
        frame_idx = np.linspace(0, video_length-1, max_length).astype(int)
        video_frames = video_frames[frame_idx]

    return video_frames

def plot_matrix_as_image_for_paper(args, matrix, names, set, text, epoch = None, run_name = None):
    # Create a figure and axis
    # only keep 2 decimal points

    matrix = np.array(matrix)
    m_min = matrix.min()
    m_max = matrix.max()

    if m_max == m_min:
        matrix= np.zeros_like(matrix)
    else:
        matrix = (matrix - m_min) / (m_max - m_min)

    # keep 2 digit first 2 digit after decimal point {val:.2f}
    matrix = np.round(matrix, 2)
    # fig, ax = plt.subplots(figsize=(len(matrix), len(matrix)))
    fig, ax = plt.subplots(figsize=(len(matrix) * 1.25, len(matrix) * 1))

    cax = ax.matshow(matrix, cmap="Blues", interpolation="nearest")  # originally was viridis

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    wandb.log({f"confusion_matrix/{set}_confusion_matrix_Rewind": wandb.Image(fig, caption=f"Epoch {epoch}")})
    if args.pdf:
        folder_name = run_name or "default"
        base_dir = getattr(args, "confusion_matrix_dir", "confusion_matrix_for_paper")
        run_dir = os.path.join(base_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)
        pdf_path = os.path.join(run_dir, f"confusion_matrix_{set}_epoch_{epoch}.pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory



def plot_confusion_matrix(h5_file, set, rewind_model, args, epoch = None, run_name = None):
    device = next(rewind_model.parameters()).device

    keys = list(h5_file.keys())
    eval_envs = keys

    text_embeddings = []
    text_list = []
    for key in eval_envs:
        embedding = np.asarray(h5_file[key]["minilm_lang_embedding"])[0].reshape(1, -1)

        text_embeddings.append(embedding)
        text_list.append(key)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    text_embeddings = torch.from_numpy(text_embeddings).to(device).float()

    pred_org_progress_list = []

    for i  in tqdm(range(len(eval_envs))):
        env = eval_envs[i]
        choose_keys = list(h5_file[env].keys())
        choose_keys = [key for key in choose_keys if "lang" not in key]

        traj_list = []
        for key in choose_keys:
            video_embedding = np.asarray(h5_file[env][key])
            if args.subsample_video:
                video_embedding = padding_video(video_embedding, args.max_length)
            traj_list.append(video_embedding)
        traj_data_all = np.stack(traj_list, axis=0)
        traj_data_all = torch.from_numpy(traj_data_all).to(device).float()

        progress_org_list = []
        for id in range(traj_data_all.shape[0]):
            traj_data = traj_data_all[id].unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1)
            pred_class = rewind_model(traj_data, text_embeddings)

            pred_class = pred_class[:, -1].squeeze()
            progress_org_list.append(pred_class.clone().cpu().detach().numpy())

        progress_org_list = np.stack(progress_org_list, axis=0)
        progress_org_list = np.mean(progress_org_list, axis=0)
        pred_org_progress_list.append(progress_org_list)

    plot_matrix_as_image_for_paper(args, pred_org_progress_list, eval_envs, set, text_list, epoch = epoch, run_name = run_name)


