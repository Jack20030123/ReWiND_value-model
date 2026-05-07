import os
import json
import h5py
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.eval_utils import compute_metrics_multi
from model import ReWiNDTransformer 
from dataset import ReWiNDVideoDataset

from utils.update_utils import train_step_fn, CosineWithMinLRScheduler
from utils.eval_confusion_matrix import plot_confusion_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "False"
DEFAULT_SCRATCH_ROOT = os.environ.get("REWIND_SCRATCH_DIR", "/scratch1/haobaizh/rewind_valuemodel")


def setup_wandb_runtime_dirs(wandb_dir):
    scratch_defaults = {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(DEFAULT_SCRATCH_ROOT, ".cache", "wandb"),
        "WANDB_CONFIG_DIR": os.path.join(DEFAULT_SCRATCH_ROOT, ".config", "wandb"),
    }
    home_defaults = {
        "WANDB_DIR": os.path.join(os.path.expanduser("~"), "wandb"),
        "WANDB_CACHE_DIR": os.path.join(os.path.expanduser("~"), ".cache", "wandb"),
        "WANDB_CONFIG_DIR": os.path.join(os.path.expanduser("~"), ".config", "wandb"),
    }
    for key, scratch_path in scratch_defaults.items():
        current = os.environ.get(key)
        home_default = home_defaults[key]
        if current and os.path.abspath(current) != os.path.abspath(home_default):
            continue
        os.makedirs(scratch_path, exist_ok=True)
        os.environ[key] = scratch_path


def resolve_scratch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(DEFAULT_SCRATCH_ROOT, path)



def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.checkpoint_dir = resolve_scratch_path(args.checkpoint_dir)
    args.wandb_dir = resolve_scratch_path(args.wandb_dir)
    args.confusion_matrix_dir = resolve_scratch_path(args.confusion_matrix_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_dim = 768
    text_dim = 384

    # TODO: Set your own WandB entity and project name
    WANDB_ENTITY_NAME = args.wandb_entity
    WANDB_PROJECT_NAME = args.wandb_project
    if args.wandb_dir:
        setup_wandb_runtime_dirs(args.wandb_dir)

    experiment_name = "ReWiND_Release_" + str(args.extra_data_type)
    if args.use_exponential_progress:
        experiment_name += f"_exp_beta{args.exponential_beta:g}"

    group_name = "ReWind_Release_" + args.extra_data_type 
    if args.use_exponential_progress:
        group_name += "_exponential_progress"
    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=group_name,
        config=args,
        name=experiment_name,
    )

    h5_train_eval_file = os.path.join(args.h5_folder_path, "metaworld_embeddings_train.h5")
    h5_eval_file = os.path.join(args.h5_folder_path, "metaworld_embeddings_eval.h5")
    h5_train_eval_file = h5py.File(h5_train_eval_file, "r")
    h5_eval_file = h5py.File(h5_eval_file, "r")
    h5_close_success_file = "datasets/metaworld_dino_embeddings_eval_close_succ.h5"
    h5_all_fail_file = "datasets/metaworld_dino_embeddings_eval_all_fail.h5"
    h5_close_success_file = h5py.File(h5_close_success_file, "r")
    h5_all_fail_file = h5py.File(h5_all_fail_file, "r")
    task_list = "utils/new_task_v2.json"
    task_list = json.load(open(task_list, "r"))
    
    openx_h5_file = h5py.File(args.openx_embedding_path, "r")
    openx_dataset = ReWiNDVideoDataset(args, openx_h5_file, sample_neg=False)
    extra_dataset = ReWiNDVideoDataset(args, h5_train_eval_file, sample_neg=True)
    
    openx_batch_size = int(round(args.batch_size * (1 - args.extra_data_ratio)))
    extra_batch_size = int(round(args.batch_size * args.extra_data_ratio))

    openx_dataloader = DataLoader(openx_dataset, batch_size=openx_batch_size, shuffle=True, num_workers=int(args.worker * 4), drop_last=True, pin_memory=False)
    extra_dataloader = DataLoader(extra_dataset, batch_size=extra_batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=False)

    rewind_model = ReWiNDTransformer(
        args=args,
        video_dim=video_dim,  # Original video embedding dimension
        text_dim=text_dim,   # Original text embedding dimension
        hidden_dim=512  # Common dimension for transformer processing
    ).to(device)


    print(rewind_model)
    base_optimizer = torch.optim.Adam(rewind_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineWithMinLRScheduler(base_optimizer, max_steps=300000, max_lr=args.lr, min_lr=1e-5)

    print("Starting training")

    for epoch in range(args.epochs):

        rewind_model.train()

        training_loader = zip(openx_dataloader, extra_dataloader)

        for batch in tqdm(training_loader, total=min(len(openx_dataloader), len(extra_dataloader)), desc=f"Epoch {epoch + 1}/{args.epochs}"):
            train_step_fn(
                args=args,
                batch=batch,
                rewind_model=rewind_model,
                optimizer=base_optimizer,
                scheduler=scheduler
            )

        rewind_model.eval()
        with torch.no_grad():
            if args.extra_data_type == "metaworld":

                plot_confusion_matrix(h5_file = h5_train_eval_file, set = "train", rewind_model = rewind_model, args = args, epoch = epoch, run_name = experiment_name)
                plot_confusion_matrix(h5_file = h5_eval_file, set = "eval", rewind_model = rewind_model, args = args, epoch = epoch, run_name = experiment_name)
        
        if epoch <= 15:
            if (epoch + 1) % args.eval_interval == 0: # too save time, we evaluate every 5 epochs
                compute_metrics_multi(args, 
                                    rewind_model, 
                                    gt_data = h5_eval_file,
                                    close_success_data=h5_close_success_file,
                                    all_fail_data=h5_all_fail_file,
                                    task_list=task_list,
                                    epoch=epoch)
        else:
            compute_metrics_multi(args, 
                                rewind_model, 
                                gt_data = h5_eval_file,
                                close_success_data=h5_close_success_file,
                                all_fail_data=h5_all_fail_file,
                                task_list=task_list,
                                epoch=epoch)
        
        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_dict = {
            "args": args,
            "model_state_dict": rewind_model.state_dict(),
            "optimizer_state_dict": base_optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        ckpt_tag = args.extra_data_type
        if args.use_exponential_progress:
            ckpt_tag += f"_exp_beta{args.exponential_beta:g}"
        checkpoint_path = os.path.join(args.checkpoint_dir, f"rewind_{ckpt_tag}_epoch_{epoch}.pth")
        torch.save(save_dict, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        


        rewind_model.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wandb_entity', type=str, required=True, help="Your WandB entity name")
    argparser.add_argument('--wandb_project', type=str, default='rewind-reward-training', help="WandB project name")
    argparser.add_argument('--h5_folder_path', type=str, default='datasets')
    argparser.add_argument('--openx_embedding_path', type=str, default='datasets/full_openx_embeddings_v2_train.h5', help="Path to the OpenX embeddings file")
    argparser.add_argument('--extra_data_type', type=str, choices=["metaworld"], default="metaworld")
    argparser.add_argument('--batch_size', type=int, default=1024)
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--worker', type=int, default=1)
    argparser.add_argument('--rewind', action='store_true')
    argparser.add_argument('--subsample_video', action='store_true')
    argparser.add_argument('--max_length', type=int, default=16)
    argparser.add_argument('--cosine_scheduler', action='store_true')
    argparser.add_argument('--clip_grad', action='store_true')
    argparser.add_argument('--extra_data_ratio', type=float, default=0.2)
    argparser.add_argument('--eval_interval', type=int, default=1)
    argparser.add_argument('--rewind_ratio', type=float, default=0.8)
    argparser.add_argument('--checkpoint_dir', type=str,
                           default=os.path.join(DEFAULT_SCRATCH_ROOT, 'checkpoints'),
                           help="Directory for newly saved ReWiND reward checkpoints")
    argparser.add_argument('--wandb_dir', type=str,
                           default=os.path.join(DEFAULT_SCRATCH_ROOT, 'wandb'),
                           help="Directory for local WandB files")
    argparser.add_argument('--confusion_matrix_dir', type=str,
                           default=os.path.join(DEFAULT_SCRATCH_ROOT, 'confusion_matrix_for_paper'),
                           help="Directory for optional confusion-matrix PDFs")
    argparser.add_argument('--use_exponential_progress', action='store_true',
                           help="Train on normalized exponential progress labels instead of linear t / H labels")
    argparser.add_argument('--exponential_beta', type=float, default=2.0,
                           help="Beta for normalized exponential progress labels")
    argparser.add_argument('--pdf', action='store_true', help="Whether to save confusion matrix as PDF")
    args = argparser.parse_args()
    main(args)
