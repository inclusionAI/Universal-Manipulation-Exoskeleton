import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from ume.tools.helpers import to_cuda, dict_unpack
from ume.tools.torch3dtransform import T_to_pose9d, pose9d_to_T
from ume.learning.model.act.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ImageNormalizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), requires_grad=False)

    def normalize(self, x):
        return (x - self.mean) / self.std


class GaussianNormalizer(nn.Module):

    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('std', std)
    
    def action_dim(self):
        return self.mu.shape[0]

    def normalize(self, x):
        return (x - self.mu) / self.std

    def denormalize(self, x):
        return x * self.std + self.mu

class ACT_UMI(nn.Module):
    
    def __init__(self, model, chunk_size, obs_dim, n_obs_steps, action_dim, hidden_dim):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.img_normalizer = ImageNormalizer()
        self.state_normalizer = GaussianNormalizer(torch.zeros(obs_dim, dtype=torch.float32), torch.ones(obs_dim, dtype=torch.float32))
        self.action_normalizer = GaussianNormalizer(torch.zeros(action_dim, dtype=torch.float32), torch.ones(action_dim, dtype=torch.float32))
        self.action_dim = action_dim
        self.obs_lowdim_map = nn.Linear(n_obs_steps * obs_dim, hidden_dim)
        self.action_map = nn.Linear(hidden_dim, action_dim)
    
    def reset(self):
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build_model_and_optimizer(cls, n_obs_steps, action_chunk_size, seed):
        # TODO temporary solution for obs_dim and action_dim
        # TODO this should be arguments
        obs_dim = 18      # umi baseline: interpose left_in_right rot6d, pos3d, right_in_left rot6d pos3d, total 18
        action_dim = 23   # openarm action: base_xyt3d[0:3], left_rot6d[3:9], left_pos3d[9:12], right_rot6d[12:18], right_pos3d[18:21], gripper_left1d[21], gripper_right1d[22], total: 23
        hidden_dim = 256
        camera_names = []
        # for i in range(n_obs_steps):
        #     camera_names.append(f"agentview_image_{i}")
        # for i in range(n_obs_steps):
        #     camera_names.append(f"robot0_eye_in_hand_image_{i}")
        # temporary solution for using openarm cameras
        for i in range(n_obs_steps):
            camera_names.append(f"camera_wrist_left_{i}")
        for i in range(n_obs_steps):
            camera_names.append(f"camera_wrist_right_{i}")
        for i in range(n_obs_steps):
            camera_names.append(f"camera_head_{i}")
        config = dict(
            seed=seed,
            camera_names=camera_names,
            num_queries=action_chunk_size,
            state_dim=hidden_dim
        )
        model, optimizer = build_ACT_model_and_optimizer(config)
        return cls(
            model, 
            chunk_size=action_chunk_size, 
            obs_dim=obs_dim,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim, 
            hidden_dim=hidden_dim
        ), optimizer

    def get_obs(cls, batch_data):
        img = torch.cat([
            # batch_data["obs"]["agentview_image"], 
            # batch_data["obs"]["robot0_eye_in_hand_image"]
            batch_data["obs"]["camera_wrist_left"],
            batch_data["obs"]["camera_wrist_right"],
            batch_data["obs"]["camera_head"]
        ], dim=1)  # (B, n_obs_steps*2, C, H, W)
        B = img.shape[0]
        obs_left_in_right_pose9d = T_to_pose9d(batch_data["obs_left_in_right_pose"]) # (B, 1, 9)
        obs_right_in_left_pose9d = T_to_pose9d(batch_data["obs_right_in_left_pose"]) # (B, 1, 9)
        obs_state = torch.cat([obs_left_in_right_pose9d, obs_right_in_left_pose9d], dim=-1).float() # (B, 1, 18)
        return img, obs_state

    def get_action(cls, batch_data):
        action_base = batch_data["action_base"]
        action_left_relative_pose9d = T_to_pose9d(batch_data["action_left_hand_relative"])
        action_right_relative_pose9d = T_to_pose9d(batch_data["action_right_hand_relative"])
        action_gripper_left_right = batch_data["action_gripper"]
        action = torch.cat([
            action_base,
            action_left_relative_pose9d,
            action_right_relative_pose9d,
            action_gripper_left_right
        ], dim=-1)
        return action
    
    def forward(self, img, qpos):
        a_hat, _, (_, _) = self.model(qpos, img)
        return a_hat

    @torch.no_grad()
    def compute_normalizer(self, dataloader):
        print("Computing state and action normalizers...")
        all_lowdim_obs = []
        all_actions = []
        for batch_data in tqdm.tqdm(dataloader):
            batch_data = to_cuda(batch_data, non_blocking=True)
            img, qpos = self.get_obs(batch_data)
            actions = self.get_action(batch_data)  # (B, action_chunk_size, action_dim)
            B, TO, O = qpos.shape
            B, TA, A = actions.shape
            all_lowdim_obs.append(qpos)
            all_actions.append(actions)
        all_lowdim_obs = torch.cat(all_lowdim_obs, dim=0).reshape(-1, O)
        all_actions = torch.cat(all_actions, dim=0).reshape(-1, A)
        obs_mu = torch.mean(all_lowdim_obs, dim=0)
        obs_std = torch.std(all_lowdim_obs, dim=0) + 1e-6
        action_mu = torch.mean(all_actions, dim=0)
        action_std = torch.std(all_actions, dim=0) + 1e-6
        # do not normalize interpose left in right rot6d
        obs_mu[0:6] = 0
        obs_std[0:6] = 1
        # do not normalize interpose right in left rot6d
        obs_mu[9: 9+6] = 0
        obs_std[9: 9+6] = 1
        # do not normalize left hand rot6d
        action_mu[3: 3+6] = 0
        action_std[3: 3+6] = 1
        # do not normalize right hand rot6d
        action_mu[12: 12+6] = 0
        action_std[12: 12+6] = 1
        print("Computed obs mean:", obs_mu)
        print("Computed obs std:", obs_std)
        print("Computed action mean:", action_mu)
        print("Computed action std:", action_std)
        self.state_normalizer = GaussianNormalizer(obs_mu, obs_std)
        self.action_normalizer = GaussianNormalizer(action_mu, action_std)

    def compute_loss(self, batch_data):
        img, qpos = self.get_obs(batch_data)
        actions = self.get_action(batch_data)  # (B, action_chunk_size, action_dim)

        normalized_img = self.img_normalizer.normalize(img)
        normalized_qpos = self.state_normalizer.normalize(qpos)
        normalized_qpos = normalized_qpos.view(normalized_qpos.shape[0], -1)  # (B, n_obs_steps*state_dim)
        normalized_actions = self.action_normalizer.normalize(actions)
        
        obs_lowdim = self.obs_lowdim_map(normalized_qpos)  # (B, hidden_dim)
        a_hat_premap = self(normalized_img, obs_lowdim)  # (B, chunk_size, action_dim)
        a_hat = self.action_map(a_hat_premap)  # (B, chunk_size, action_dim)
        
        # print(normalized_img.shape, obs_lowdim.shape, normalized_qpos.shape)
        # print(a_hat.shape, normalized_actions.shape)
        
        loss = F.l1_loss(a_hat, normalized_actions)
        stats_dict = dict()
        stats_dict["loss"] = loss.item()
        return loss, stats_dict

    @torch.no_grad()
    def predict_action(self, batch_data):
        img, qpos = self.get_obs(batch_data)
        normalized_img = self.img_normalizer.normalize(img)
        normalized_qpos = self.state_normalizer.normalize(qpos)
        normalized_qpos = normalized_qpos.view(normalized_qpos.shape[0], -1)  # (B, n_obs_steps*state_dim)
        obs_lowdim = self.obs_lowdim_map(normalized_qpos)  # (B, hidden_dim)
        a_hat_premap = self(normalized_img, obs_lowdim)  # (B, chunk_size, action_dim)
        a_hat = self.action_map(a_hat_premap)  # (B, chunk_size, action_dim)
        a_hat = self.action_normalizer.denormalize(a_hat)
        
        # openarm_left_action = a_hat[..., :8]
        # openarm_right_action = a_hat[..., 8:16]
        # base_action = a_hat[..., 16:]
        action_base = a_hat[..., :3]
        action_left_hand_relative = a_hat[..., 3: 3+9]
        action_right_hand_relative = a_hat[..., 3+9: 3+9+9]
        action_gripper = a_hat[..., 3+9+9:]
        
        action_left_hand_relative_T4x4 = pose9d_to_T(action_left_hand_relative)
        action_right_hand_relative_T4x4 = pose9d_to_T(action_right_hand_relative)
        
        action_dict = dict(
            action_base=action_base.cpu().numpy(),
            action_left_hand_relative=action_left_hand_relative_T4x4.cpu().numpy(),
            action_right_hand_relative=action_right_hand_relative_T4x4.cpu().numpy(),
            action_gripper=action_gripper.cpu().numpy()
        )
        return action_dict


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    act_model, optimizer = ACT_UMI.build_model_and_optimizer(
        n_obs_steps=2,
        action_chunk_size=16,
        seed=0
    )
    parameter_count = count_parameters(act_model)
    print("ACT_Abs model parameters:", parameter_count)
    print("ACT_Abs model parameters (M):", parameter_count / 1_000_000)