
import argparse
import csv
import math
import time
from os.path import join

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st
import torch
import numpy as np
import yaml
import wandb
from torch.utils.data import Dataset

from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits


class SbiDataset(Dataset):
    def __init__(self, theta, x):
        super(SbiDataset, self).__init__()

        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


def generate_dataset(settings, batch_size=1, directory_save=None):
    task = sbibm.get_task(settings["task"]["name"])
    prior = task.get_prior()
    simulator = task.get_simulator()
    num_train_samples = settings["task"]["num_train_samples"]
    nr_batches = math.ceil(num_train_samples / batch_size)
    theta = []
    x = []
    for _ in range(nr_batches):
        theta_sample = prior(batch_size)
        x_sample = simulator(theta_sample)
        theta.append(theta_sample)
        x.append(x_sample)
    x = np.vstack(x)[:num_train_samples]
    theta = np.vstack(theta)[:num_train_samples]
    if directory_save is not None:
        np.save(join(directory_save, 'x.npy'), x)
        np.save(join(directory_save, 'theta.npy'), theta)

    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)
    settings["task"]["dim_theta"] = theta.shape[1]
    settings["task"]["dim_x"] = x.shape[1]

    dataset = SbiDataset(theta, x)
    return dataset


def load_dataset(directory_save, settings):
    x = np.load(join(directory_save, 'x.npy'))
    theta = np.load(join(directory_save, 'theta.npy'))

    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)
    settings["task"]["dim_theta"] = theta.shape[1]
    settings["task"]["dim_x"] = x.shape[1]

    dataset = SbiDataset(theta, x)
    return dataset



def train_model(train_dir, settings, train_loader, test_loader, use_wandb=False):
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )

    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device=settings["training"].get("device", "cpu"),
    )

    # Before training you need to call the following lines:
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        early_stopping=True,
        use_wandb=use_wandb,
    )

    # load the best model
    best_model = build_model_from_kwargs(
        filename=join(train_dir, "best_model.pt"),
        device=settings["training"].get("device", "cpu"),
    )
    return best_model


def evaluate_model(train_dir, settings, dataset, model, use_wandb=False, compute_c2st=False):
    task = sbibm.get_task(settings["task"]["name"])

    c2st_scores = {}
    flow_times = []  # Track sampling times
    
    print("Generating posterior samples and plots...")
    for obs in range(1, 11):
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = 1000  # Generate 1000 teacher samples for evaluation
        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        # Time flow sampling
        start_time = time.time()
        
        # generate (num_samples * 2), to account for samples outside of the prior
        posterior_samples = model.sample_batch(observation.repeat((num_samples * 2, 1)))
        
        flow_time = time.time() - start_time
        flow_times.append(flow_time)
        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )

        # discard samples outside the prior
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        print(
            f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% of the samples "
            f"lie outside of the prior. Discarding these."
        )
        posterior_samples = posterior_samples[prior_mask]

        n = min(len(reference_samples), len(posterior_samples))
        
        # Only compute C2ST if requested
        if compute_c2st:
            c2st_score = c2st(posterior_samples[:n], reference_samples[:n])
            c2st_scores[f"C2ST {obs}"] = c2st_score.item()
            print(f"C2ST score for observation {obs}: {c2st_score.item():.3f}")
            title = f"Observation {obs}: C2ST = {c2st_score.item():.3f}, Time = {flow_time:.3f}s ({num_samples*2} generated)"
        else:
            title = f"Observation {obs}: Time = {flow_time:.3f}s ({num_samples*2} generated)"
        
        # Always generate plots
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            posterior_samples[:, 0],
            posterior_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label=f"flow matching ({len(posterior_samples)} samples)",
        )
        plt.scatter(
            reference_samples[:, 0],
            reference_samples[:, 1],
            s=0.5,
            alpha=0.2,
            label=f"reference ({len(reference_samples)} samples)",
        )
        plt.title(title)
        plt.legend()
        plt.savefig(join(train_dir, f"posterior_{obs}.png"))
        plt.close()  # Close figure to prevent memory buildup
    
    if not compute_c2st:
        print("Skipped C2ST computation (use --compute_c2st to enable)")
    
    # Speed summary and plot for flow model
    avg_time = np.mean(flow_times)
    print(f"Average flow matching sampling time: {avg_time:.3f} ± {np.std(flow_times):.3f} seconds")
    
    # Speed comparison plot with sample information
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), flow_times, 'ro-', label='Flow Matching Sampling Time')
    
    # Add sample count annotations
    for i, (obs, time_val) in enumerate(zip(range(1, 11), flow_times)):
        # We generate num_samples * 2 = 1000 * 2 = 2000 samples
        sample_count = 1000 * 2  # Actual number of samples generated
        ax.annotate(f'{sample_count}', (obs, time_val), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Observation Number')
    ax.set_ylabel('Sampling Time (seconds)')
    ax.set_title('Flow Matching Sampling Speed (numbers show sample count generated)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(join(train_dir, "sampling_speed.png"), dpi=150, bbox_inches='tight')
    plt.close()

    with open(
        join(train_dir, "c2st.csv"), "w"
    ) as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, c2st_scores.keys())
        w.writeheader()
        w.writerow(c2st_scores)

    if use_wandb:
        wandb.log(c2st_scores)
        wandb.log({"posteriors": wandb.Image(join(train_dir, "posteriors.png"))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Base save directory for the evaluation"
    )
    parser.add_argument(
        "--compute_c2st", action="store_true", default=False,
        help="Compute C2ST scores (slow, default: False)"
    )

    args = parser.parse_args()

    with open(join(args.train_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    use_wandb = settings["training"].get("wandb")
    if use_wandb:
        import wandb

        wandb.init(config=settings, dir=args.train_dir, **settings["training"]["wandb"])

    dataset = generate_dataset(settings)

    train_loader, test_loader = build_train_and_test_loaders(
        dataset,
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )

    model = train_model(
        args.train_dir,
        settings=settings,
        train_loader=train_loader,
        test_loader=test_loader,
        use_wandb=use_wandb,
    )
    evaluate_model(args.train_dir, settings, dataset, model, use_wandb=False, compute_c2st=False)
