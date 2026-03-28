import optuna

from utils.tuner import Tuner
from utils.args import args

tuner = Tuner(
    script=args.script,
    metric=args.metric,
    metric_last_n_average_window=args.metric_last_n_average_window,
    direction=args.direction,
    aggregation_type=args.aggregation_type,
    target_scores=args.target_scores,
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", args.learning_rate_o["min"], args.learning_rate_o["max"], log=args.learning_rate_o["log"]),
        "num-minibatches": trial.suggest_categorical("num-minibatches", args.num_minibatches_o),
        "update-epochs": trial.suggest_categorical("update-epochs", args.update_epochs_o),
        "num-steps": trial.suggest_categorical("num-steps", args.num_steps_o),
        "vf-coef": trial.suggest_float("vf-coef", args.vf_coef_o["min"], args.vf_coef_o["max"], log=args.vf_coef_o["log"]),
        "max-grad-norm": trial.suggest_float("max-grad-norm", args.max_grad_norm_o["min"], args.max_grad_norm_o["max"], log=args.max_grad_norm_o["log"]),
        "total-timesteps": args.total_timesteps_per_trial,
        "num-envs": args.num_envs,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    storage=args.storage
)

args.force_reload = True

tuner.tune(
    num_trials=args.num_trials,
    num_seeds=args.num_seeds,
)