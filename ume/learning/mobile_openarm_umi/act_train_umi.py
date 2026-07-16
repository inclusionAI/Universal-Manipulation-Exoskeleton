import os
import time
import tqdm
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import wandb  # ADDED
from ume.tools.helpers import to_cuda
from ume.learning.mobile_openarm_umi.model import ACT_UMI
from ume.learning.mobile_openarm_umi.openarm_umi_dataset import OpenArmUMIDataset
import datetime
from time import strftime, localtime

# os.environ["WANDB_MODE"] = "offline"

def get_datetime_str(timestamp: float) -> str:
    timestamp = float(timestamp)
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    ts_dir = f"{strftime('%Y_%m_%d_%H_%M_%S', localtime(timestamp))}_{timezone}"
    return ts_dir

def get_exp_dir(timestamp: float) -> str:
    return f"data/model_ckpts/{get_datetime_str(timestamp)}"


def save_checkpoint(path, info, model, optimizer):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
    with open(os.path.join(path, "info.yaml"), "w") as f:
        yaml.dump(info, f)


if __name__ == "__main__":
    
    CONFIG = dict(
        experiment_name="umi",
        seed=42,
        exp_dir=get_exp_dir(time.time()),
        dataset_path='data/collecting',
        # model parameters
        action_horizon=20,
        action_duration=2.0,  # seconds
        obs_horizon=1,
        obs_duration=1.0,  # seconds
        # training parameters
        batch_size=200,
        # per_worker_fetch_size=64,
        save_every_n_steps=5000,
    )
    
    os.makedirs(CONFIG["exp_dir"], exist_ok=True)
    print(f"Experiment directory: {CONFIG['exp_dir']} created.")
    
    wandb.init(project="act-openarm", name=os.path.basename(CONFIG["exp_dir"]), config=CONFIG)  # ADDED

    dataset = OpenArmUMIDataset(
        # shape_meta=SHAPE_META,
        data_root=CONFIG['dataset_path'], 
        action_horizon=CONFIG['action_horizon'],
        obs_horizon=CONFIG['obs_horizon'],
        action_duration=CONFIG['action_duration'],
        obs_duration=CONFIG['obs_duration'],
        camera_names=["camera_wrist_left", "camera_head", "camera_wrist_right"]
    )


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        prefetch_factor=8,
        shuffle=True,
        num_workers=24,
        persistent_workers=True,
        pin_memory=True
    )

    act_model, optimizer = ACT_UMI.build_model_and_optimizer(
        n_obs_steps=CONFIG['obs_horizon'],
        action_chunk_size=CONFIG["action_horizon"],
        seed=CONFIG["seed"]
    )
    act_model.compute_normalizer(dataloader)
    act_model.cuda()
    act_model.train()

    # train loop
    epoch_iter_idx = 0
    train_step_idx = 0
    last_epoch_loss = 0
    t_train_loop_start = time.time()
    while True:
        t_epoch_start = time.time()
        progress_bar = tqdm.tqdm(iterable=dataloader, total=len(dataloader), desc=f"Epoch {epoch_iter_idx} training")
        
        save_checkpoint(
            path=os.path.join(CONFIG["exp_dir"], f"ckpt_latest"),
            info=dict(
                start_time=t_train_loop_start,
                time_now=time.time(),
                elapsed_time=time.time() - t_train_loop_start,
                epoch_idx=epoch_iter_idx,
                train_step_idx=train_step_idx,
                config=CONFIG,
                epoch_loss=last_epoch_loss,
                datetime_now=get_datetime_str(time.time())
            ),
            model=act_model, optimizer=optimizer
        )
        
        epoch_loss = []
        for batch_idx, batch_data in enumerate(progress_bar):
            
            # prepare batch data
            batch_data = to_cuda(batch_data, non_blocking=True)

            # save model checkpoint
            if train_step_idx % CONFIG["save_every_n_steps"] == 0:
                save_checkpoint(
                    path=os.path.join(CONFIG["exp_dir"], f"ckpt_step_{train_step_idx}"), 
                    info=dict(
                        start_time=t_train_loop_start,
                        time_now=time.time(),
                        elapsed_time=time.time() - t_train_loop_start,
                        epoch_idx=epoch_iter_idx,
                        train_step_idx=train_step_idx,
                        config=CONFIG,
                        epoch_loss=last_epoch_loss,
                        datetime_now=get_datetime_str(time.time())
                    ),
                    model=act_model, optimizer=optimizer
                )

            # training step
            optimizer.zero_grad()
            loss, stats_dict = act_model.compute_loss(batch_data)
            loss.backward()
            optimizer.step()

            # log stats
            train_step_idx += 1
            epoch_loss.append(loss.item())
            if train_step_idx % 10 == 0:
                progress_bar.set_postfix(step=train_step_idx, loss=stats_dict["loss"])
                wandb.log({"train/step_loss": stats_dict["loss"]}, step=train_step_idx)  # ADDED
        
        epoch_iter_idx += 1
        last_epoch_loss = np.mean(epoch_loss).item()
        print(f"    experiment {CONFIG['exp_dir']} took {(time.time() - t_epoch_start) / 3600} hours, epoch average loss: {last_epoch_loss:.6f}")
        wandb.log({"train/epoch_loss": last_epoch_loss, "train/epoch": epoch_iter_idx}, step=train_step_idx)  # ADDED
    