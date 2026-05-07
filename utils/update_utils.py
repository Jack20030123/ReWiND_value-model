import torch
import wandb
import math
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]

def train_step_fn(args, batch, rewind_model, optimizer, scheduler):
    #set to cuda
    openx_data, extra_data = batch
    openx_len = len(openx_data["video_array"])
    extra_len = len(extra_data["video_array"])

    rewind_model.train()
    optimizer.zero_grad()
    positive_video_array = torch.cat([openx_data["video_array"], extra_data["video_array"]], dim = 0).to(device).float()
    positive_text_array = torch.cat([openx_data["text_array"].squeeze(1), extra_data["text_array"].squeeze()], dim = 0).to(device).float()              
    positive_progress = torch.cat([openx_data["progress"], extra_data["progress"]], dim = 0).to(device)

    negative_video_array_1 = torch.roll(positive_video_array, extra_len, 0)
    negative_text_array_1 = positive_text_array.clone()
    negative_progress_1 = torch.zeros_like(positive_progress)

    openx_pos_video_array = torch.cat([positive_video_array[:openx_len], negative_video_array_1[:openx_len]], dim = 0)
    openx_pos_text_array = torch.cat([positive_text_array[:openx_len], negative_text_array_1[:openx_len]], dim = 0)
    openx_pos_progress = torch.cat([positive_progress[:openx_len], negative_progress_1[:openx_len]], dim = 0)
        
    extra_pos_video_array = torch.cat([positive_video_array[openx_len:], negative_video_array_1[openx_len:]], dim = 0)
    extra_pos_text_array = torch.cat([positive_text_array[openx_len:], negative_text_array_1[openx_len:]], dim = 0)
    extra_pos_progress = torch.cat([positive_progress[openx_len:], negative_progress_1[openx_len:]], dim = 0)

    video_array = torch.cat([openx_pos_video_array, extra_pos_video_array], dim = 0)
    text_array = torch.cat([openx_pos_text_array, extra_pos_text_array], dim = 0)
    progress = torch.cat([openx_pos_progress, extra_pos_progress], dim = 0).float()

    openx_len = len(openx_pos_video_array)
    extra_len = len(extra_pos_video_array)

    video_embedding = video_array

    # Binary classification targets
    compressed_extra_class_label = extra_data["class_label"][:, 0].float()
    openx_target = torch.cat([torch.ones(openx_len // 2), torch.zeros(openx_len // 2)], dim=0).to(device)
    extra_target = torch.cat([compressed_extra_class_label, torch.zeros(extra_len // 2)], dim=0).to(device)

    # Get predictions from classifier
    progress_pred = rewind_model(video_embedding, text_array)

    openx_progress_pred = progress_pred[:openx_len]
    extra_progress_pred = progress_pred[openx_len:]

    openx_progress_target = progress[:openx_len]
    extra_progress_target = progress[openx_len:]

    valid_openx_progress_pred = openx_progress_pred[openx_target.bool()]
    valid_openx_progress_target = openx_progress_target[openx_target.bool()]

    valid_extra_progress_pred = extra_progress_pred[extra_target.bool()]
    valid_extra_progress_target = extra_progress_target[extra_target.bool()]

    rest_openx_progress_pred = openx_progress_pred[~openx_target.bool()]
    rest_openx_progress_target = openx_progress_target[~openx_target.bool()]

    rest_extra_progress_pred = extra_progress_pred[~extra_target.bool()]
    rest_extra_progress_target = extra_progress_target[~extra_target.bool()]

    if getattr(args, "use_pairwise_progress_diff_loss", False):
        openx_progress_loss = pairwise_progress_diff_loss(valid_openx_progress_pred, valid_openx_progress_target)
        extra_progress_loss = pairwise_progress_diff_loss(valid_extra_progress_pred, valid_extra_progress_target)
        rest_openx_progress_loss = pairwise_progress_diff_loss(rest_openx_progress_pred, rest_openx_progress_target)
        rest_extra_progress_loss = pairwise_progress_diff_loss(rest_extra_progress_pred, rest_extra_progress_target)
    else:
        openx_progress_loss = mse_loss(valid_openx_progress_pred[:,1:].squeeze(), valid_openx_progress_target[:,1:])
        extra_progress_loss = mse_loss(valid_extra_progress_pred[:,1:].squeeze(), valid_extra_progress_target[:,1:])
        rest_openx_progress_loss = mse_loss(rest_openx_progress_pred[:,1:].squeeze(), rest_openx_progress_target[:,1:])
        rest_extra_progress_loss = mse_loss(rest_extra_progress_pred[:,1:].squeeze(), rest_extra_progress_target[:,1:])

    total_len = len(openx_progress_pred) + len(extra_progress_pred) + len(rest_openx_progress_pred) + len(rest_extra_progress_pred)

    loss = openx_progress_loss * len(openx_progress_pred) / total_len \
            + extra_progress_loss * len(extra_progress_pred) / total_len \
            + rest_openx_progress_loss * len(rest_openx_progress_pred) / total_len \
            + rest_extra_progress_loss * len(rest_extra_progress_pred) / total_len

    loss.backward()
    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(rewind_model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Log all metrics
    wandb_log = {
        "train/progress_loss": loss.item(),
        "lr": optimizer.param_groups[0]["lr"],
    }
    if getattr(args, "use_pairwise_progress_diff_loss", False):
        wandb_log["train/pairwise_progress_diff_loss"] = loss.item()
    wandb.log(wandb_log)
    return loss.item()


def pairwise_progress_diff_loss(progress_pred, progress_target):
    if progress_pred.shape[0] == 0:
        return progress_pred.sum() * 0.0

    pred = progress_pred[:, 1:].squeeze(-1)
    target = progress_target[:, 1:].float()
    seq_len = pred.shape[1]
    if seq_len < 2:
        return mse_loss(pred, target)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(2)
    target_diff = target.unsqueeze(1) - target.unsqueeze(2)
    pair_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=pred.device),
        diagonal=1,
    )
    return mse_loss(pred_diff[:, pair_mask], target_diff[:, pair_mask])
