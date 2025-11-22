import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(
    title,
    path,
    train_loss,
    test_loss,
    train_acc,
    test_acc,
    eigs,
    eig_freq,
    regions_pier,
    regions_freq,
    num_samples_line,
    lr,
    regions_hanin,
    num_hanin_line_samples,
    regions_humayan,
    num_orthonormal_vectors_humayan,
    lr_schedule_gamma,
    lr_schedule_steps,
    opt,
):
    """
    Replaces save_files_final. Generates and saves plots for loss, accuracy, and sharpness.
    """

    steps = np.arange(len(train_loss))
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(steps, train_loss.cpu(), label="Train Loss")
    ax.plot(steps, test_loss.cpu(), label="Test Loss", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(steps, train_acc.cpu(), label="Train Accuracy")
    ax.plot(steps, test_acc.cpu(), label="Test Accuracy", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    if eig_freq > 0 and len(eigs) > 0:
        eig_steps = np.arange(0, len(train_loss), eig_freq)[: len(eigs)]
        ax = axes[1, 0]
        ax.plot(eig_steps, eigs.cpu())
        if lr_schedule_gamma != 1:
            for i in range(0, len(train_loss), step=lr_schedule_steps):
                ax.axhline(
                    y=2 / (lr * lr_schedule_gamma**i),
                    linestyle="--",
                    linewidth=1,
                    label=f"lr={lr * lr_schedule_gamma**i:.3f}",
                )
        else:
            if opt == "adam":
                ax.axhline(y=38 / lr, color="r", linestyle="--",
                           linewidth=1, label="38/lr")
            ax.axhline(y=2 / lr, color="r", linestyle="--",
                       linewidth=1, label="2/lr")
        ax.set_xlabel("Step")
        ax.set_ylabel("Eigenvalues (Sharpness)")
        ax.set_title("Top Hessian Eigenvalues")
        ax.grid(True)
    else:
        axes[1, 0].axis("off")

    if regions_freq > 0 and len(regions_pier) > 0:
        regions_steps = np.arange(0, len(train_loss), regions_freq)[
            : len(regions_pier)]
        ax = axes[1, 1]
        ax.plot(regions_steps, regions_pier[:, 0].cpu())
        ax.fill_between(regions_steps, regions_pier[:, 1].cpu(
        ), regions_pier[:, 2].cpu(), alpha=0.3)
        ax.set_ylim(0, num_samples_line)
        ax.set_xlabel("Step")
        ax.set_ylabel("Pier Regions")
        ax.set_title("Pier Count of linear regions")
        ax.grid(True)
    else:
        axes[1, 1].axis("off")

    if regions_freq > 0 and len(regions_humayan) > 0:
        regions_steps = np.arange(0, len(train_loss), regions_freq)[
            : len(regions_humayan)]
        ax = axes[2, 0]
        ax.plot(regions_steps, regions_humayan[:, 0].cpu())
        ax.fill_between(regions_steps, regions_humayan[:, 1].cpu(
        ), regions_humayan[:, 2].cpu(), alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Humayan Regions")
        ax.set_title("Humayan Count of linear regions")
        ax.grid(True)
    else:
        axes[2, 0].axis("off")

    if regions_freq > 0 and len(regions_hanin) > 0:
        regions_steps = np.arange(0, len(train_loss), regions_freq)[
            : len(regions_hanin)]
        ax = axes[2, 1]
        ax.plot(regions_steps, regions_hanin[:, 0].cpu())
        ax.fill_between(regions_steps, regions_hanin[:, 1].cpu(
        ), regions_hanin[:, 2].cpu(), alpha=0.3)
        ax.set_ylim(0, num_hanin_line_samples)
        ax.set_xlabel("Step")
        ax.set_ylabel("Hanin Regions")
        ax.set_title("Hanin Count of linear regions")
        ax.grid(True)
    else:
        axes[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle
    save_path = f"{path}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved training plots to {save_path}")
