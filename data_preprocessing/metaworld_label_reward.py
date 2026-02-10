import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_preprocessing.generate_dino_embeddings import DINO_BATCH_SIZE
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from model import ReWiNDTransformer 
from utils.processing_utils import dino_load_image


DINO_BATCH_SIZE = 64

video_dim = 768
text_dim = 384
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
dinov2_vits14 = dinov2_model.to(device)
    
def get_dino_embeddings(imgs_list):
    '''
    Get DINO embeddings for a list of images.
    Args:
        imgs_list: List of images (numpy arrays).
    Returns:
        Numpy array of DINO embeddings.
    '''
    episode_images_dino = [
        dino_load_image(img) for img in imgs_list
    ]
    episode_images_dino = [
            torch.concatenate(
                episode_images_dino[i : i + DINO_BATCH_SIZE]
            )
            for i in range(
                0, len(episode_images_dino), DINO_BATCH_SIZE
            )
        ]
    embedding_list = []
    for batch in episode_images_dino:
        episode_image_embeddings = (
            dinov2_vits14(batch.to(device))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        if len(episode_image_embeddings.shape) == 1:
            episode_image_embeddings = np.expand_dims(episode_image_embeddings, 0)
        embedding_list.append(episode_image_embeddings)

    episode_image_embeddings = np.concatenate(embedding_list)
    return episode_image_embeddings

def padding_video(video_frames, max_length):

    if type(video_frames) == np.ndarray:
        video_frames = torch.tensor(video_frames)

    if len(video_frames) > max_length:
        index = np.linspace(0, len(video_frames) - 1, max_length).astype(int)
        video_frames = video_frames[index]

    else:
        # padding last frame
        padding_num = max_length - len(video_frames)
        last_frame = video_frames[-1].unsqueeze(0)
        padding_frames = last_frame.repeat(padding_num, 1)
        video_frames = torch.cat([video_frames, padding_frames], dim=0)
    return video_frames

def load_rewind_model(model_path):
    '''
    Load the ReWiND model from the specified path.
    '''
    model_state_dict = torch.load(model_path, map_location=device, weights_only=False)
    args = model_state_dict["args"]
    rewind_model = ReWiNDTransformer(
        args=args,
        video_dim=video_dim, 
        text_dim=text_dim,   
        hidden_dim=512  
    ).to(device) 
    rewind_model.load_state_dict(model_state_dict["model_state_dict"])

    return args, rewind_model       


def label_trajectories_iteratively(
    args, rewind_model, traj_h5, embedding_h5
):
    """
    Processes trajectories iteratively, computes rewards, and saves data directly to the output HDF5 file.
    Args:
        args: Command line arguments.
        rewind_model: The ReWiND model for reward computation.
        traj_h5: HDF5 file containing the trajectories.
        image_keys: List of keys in the HDF5 file that correspond to image data.
    """

    training_keys = list(embedding_h5.keys())
    # compute total timesteps
    total_timesteps = 0
    num = 0
    for key in training_keys:
        for traj_id in traj_h5[key].keys():
            total_timesteps += len(traj_h5[key][traj_id]["reward"])
            num += 1
            # import pdb ;pdb.set_trace()
    total_timesteps = int(total_timesteps * 5) # we have 5 annotations per trajectory

    labeled_dataset = h5py.File(args.output_path, "w")
    labeled_dataset.create_dataset("action", (total_timesteps,4), dtype="float32")
    labeled_dataset.create_dataset("rewards", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset("done", (total_timesteps,), dtype="float32")
    labeled_dataset.create_dataset("policy_lang_embedding",(total_timesteps, 384),dtype="float32",)
    labeled_dataset.create_dataset("img_embedding", (total_timesteps, 768),dtype="float32",)
    labeled_dataset.create_dataset("env_id", (total_timesteps,), dtype="S20")

    current_timestep = 0

    for key in tqdm(training_keys):
        for traj_id in traj_h5[key].keys():

            save_actions = np.array(traj_h5[key][traj_id]["action"])
            save_dones = np.array(traj_h5[key][traj_id]["done"])

            traj_data = traj_h5[key][traj_id]
            num_steps = len(traj_data["done"])
            video_frames = np.array(traj_data["img"])
            video_frames = [img for img in video_frames]
            video_frame_embeddings = get_dino_embeddings(video_frames)
            save_video_slices = video_frame_embeddings
            video_slices = [padding_video(video_frame_embeddings[0:-i], 
                                          max_length = args.max_length) for i in range(len(video_frame_embeddings)-1, 0, -1)] + \
                                          [padding_video(video_frame_embeddings[0:], max_length = args.max_length)]
            video_slices = torch.stack(video_slices).float().to(device) # (num_steps, max_length, 768)

            # iteratively compute reward for each step
            lang_embeddings = np.array(embedding_h5[key]["minilm_lang_embedding"])
            last_index_mask = torch.zeros((video_slices.shape[0], args.max_length)).to(device).bool()
            # For each step i, mark the position of the last valid frame in the padded sequence
            # If i < max_length: last valid frame is at position i (diagonal)
            # If i >= max_length: last valid frame is at position max_length-1 (last column)
            for i in range(video_slices.shape[0]):
                last_frame_idx = min(i, args.max_length - 1)
                last_index_mask[i, last_frame_idx] = 1
            for i in range(len(lang_embeddings)):
                lang_embedding = lang_embeddings[i]
                lang_embedding = torch.tensor(lang_embedding).float().to(device)

                language_embedding = lang_embedding.unsqueeze(0).repeat(video_slices.shape[0], 1) # (num_steps, 1, 384)
                with torch.no_grad():
                    reward_outputs = rewind_model(video_slices, language_embedding).squeeze(-1) # (num_steps, 1)
                    reward_outputs = reward_outputs[last_index_mask] # (num_steps, max_length)
                    progress_values = reward_outputs.cpu().numpy()

                    if args.use_progress_diff:
                        # Progress diff: reward = P(s') - P(s)
                        save_reward_outputs = progress_values[1:] - progress_values[:-1]
                    else:
                        # Original: reward = P(s)
                        save_reward_outputs = progress_values[1:]
                save_lang_embedding = lang_embedding.repeat(num_steps, 1).cpu().numpy()

                # Save to labeled dataset
                labeled_dataset["action"][current_timestep:current_timestep+num_steps] = save_actions
                labeled_dataset["done"][current_timestep:current_timestep+num_steps] = save_dones
                labeled_dataset["rewards"][current_timestep:current_timestep+num_steps] = save_reward_outputs
                labeled_dataset["policy_lang_embedding"][current_timestep:current_timestep+num_steps] = save_lang_embedding
                labeled_dataset["img_embedding"][current_timestep:current_timestep+num_steps] = save_video_slices[:-1]
                labeled_dataset["env_id"][current_timestep:current_timestep+num_steps] = key

                current_timestep += num_steps
    print(f"Successfully processed and saved {current_timestep} timesteps.")



def main():
    parser = argparse.ArgumentParser(description="Label rewards for trajectories.")
    parser.add_argument(
        "--h5_video_path",
        default="datasets/metaworld_generation.h5",
        help="Path to the trajectories file (HDF5 format).",
    )
    parser.add_argument(
        "--h5_embedding_path",
        default="datasets/metaworld_embeddings_train.h5",
        help="To extract language annotation embeddings.",
    )
    parser.add_argument(
        "--reward_model_path",
        help="Path to the saved model.",
        default="checkpoints/rewind_metaworld_epoch_19.pth",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/metaworld_labeled.h5",
        help="Path to save the labeled dataset (HDF5 format).",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=1000000000000000000,
        help="Window length for video frame embeddings.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for encoding."
    )
    parser.add_argument(
        "--use_progress_diff",
        action="store_true",
        help="Use progress diff (P(s') - P(s)) instead of P(s) as reward."
    )

    args = parser.parse_args()


    config, rewind_model = load_rewind_model(args.reward_model_path)
    rewind_model.eval()
    rewind_model.to(device)
    print("Loaded ReWiND model.")

    h5_video_file = h5py.File(args.h5_video_path, "r")
    h5_embedding_file = h5py.File(args.h5_embedding_path, "r")
    args.max_length = config.max_length

    label_trajectories_iteratively(
        args, rewind_model, h5_video_file, h5_embedding_file
    )



if __name__ == "__main__":
    main()
