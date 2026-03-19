import os
import sys
from dataclasses import dataclass, field
import yaml

import tyro


@dataclass
class TrainArgs:
    config: str = "configs/train/default.yaml"
    """config file"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_frequency: int = 50
    """save agent every n update epochs"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def get_train_args():
    tmp_args = tyro.cli(TrainArgs, args=["--config", "configs/train/default.yaml"] if "--config" not in sys.argv else None)

    with open(tmp_args.config, "r") as f:
        conf = yaml.safe_load(f)

    args = tyro.cli(TrainArgs, default=TrainArgs(**conf))

    return args


@dataclass
class RangeOptions:
    min: float = 0
    max: float = 1
    log: bool = False


@dataclass
class TuneArgs:
    config: str = "configs/tune/default.yaml"
    """config file"""
    script: str = "train.py"
    """training script to run"""
    metric: str = "charts/episodic_return"
    """metric to use for tuning"""
    metric_last_n_average_window: int = 50
    """window over witch the metric gets averaged"""
    direction: str = "maximize"
    """direction in witch to optimize"""
    aggregation_type: str = "average"
    """aggregation type for multiple environments"""
    target_scores: dict = field(default_factory=lambda: {
        "CartPole-v1": [0, 500],
    })
    """lower and upper bound for reward per environment"""

    learning_rate: RangeOptions = field(default_factory=lambda: RangeOptions(
        min=0.0003,
        max=0.003, 
        log=True
        )
    )
    """range for learning rate tuning"""
    num_minibatches: list = field(default_factory=lambda:[1, 2, 4])
    """options for number of minibatches"""
    update_epochs: list = field(default_factory=lambda:[1, 2, 4, 8])
    """options for number of epochs per update"""
    num_steps: list = field(default_factory=lambda:[5, 16, 32, 64, 128])
    """options for number of steps to run in each environment per policy rollout"""
    vf_coef: RangeOptions = field(default_factory=lambda: RangeOptions(
        min=0, 
        max=5, 
        log=False
        )
    )
    """range for coefficient of the value function"""
    max_grad_norm: RangeOptions = field(default_factory=lambda: RangeOptions(
        min=0, 
        max=5, 
        log=False
        )
    )
    """range for the maximum norm for the gradient clipping"""
    total_timesteps: int = 100000
    """total timesteps of each trial"""
    num_envs: int = 16
    """the number of parallel game environments"""
    storage: str = "sqlite:///tune.db"
    """storage location of tuning database"""

    num_trials: int = 100
    """number of trials to optimize over"""
    num_seeds: int = 3
    """number of seeds/experiments per trial"""


def get_tune_args():
    tmp_args = tyro.cli(TuneArgs, args=["--config", "configs/tune/default.yaml"] if "--config" not in sys.argv else None)

    with open(tmp_args.config, "r") as f:
        conf = yaml.safe_load(f)

    args = tyro.cli(TuneArgs, default=TuneArgs(**conf))

    return args