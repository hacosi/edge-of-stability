import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os


def plot_training_results(
    history,
    title,
    path,
    eig_freq,
    regions_freq,
    num_samples_line,
    lr,
    num_hanin_line_samples,
    num_orthonormal_vectors_humayan,
    lr_schedule_gamma,
    lr_schedule_steps,
    opt,
):
    """
    Replaces save_files_final. Generates and saves plots for loss, accuracy, and sharpness.
    """

    train_loss = history["train_loss"]
    test_loss = history["test_loss"]
    train_acc = history["train_acc"]
    test_acc = history["test_acc"]
    eigs = history["eigs"]
    regions_pier = history["regions_pier"]
    regions_hanin = history["regions_hanin"]
    regions_humayan = history["regions_humayan"]

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
            else:
                ax.axhline(y=2 / lr, color="r", linestyle="--",
                           linewidth=1, label="2/lr")
        ax.set_xlabel("Step")
        ax.set_ylabel("Eigenvalues (Sharpness)")
        ax.legend()
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
        ax.plot(regions_steps, regions_humayan[:, 0].cpu(), label="scale 1")
        ax.plot(regions_steps, history["regions_humayan_scale_0.01"][:, 0].cpu(
        ), label="scale 0.01")
        ax.plot(regions_steps, history["regions_humayan_scale_0.1"][:, 0].cpu(
        ), label="scale 0.1")
        ax.plot(regions_steps, history["regions_humayan_scale_0.5"][:, 0].cpu(
        ), label="scale 0.5")
        ax.plot(regions_steps,
                history["regions_humayan_scale_10"][:, 0].cpu(), label="scale 10")
        ax.fill_between(regions_steps, regions_humayan[:, 1].cpu(
        ), regions_humayan[:, 2].cpu(), alpha=0.3)
        ax.legend()
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


def make_live_animation(
    history,
    opt,
    lr,
    eig_freq,
    regions_freq,
    num_samples_line,
    num_hanin_line_samples,
    num_humayan_orthonormal_vectors,
    path,
    fps=4,
):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    ax_loss = axes[0, 0]
    ax_acc = axes[0, 1]
    ax_sharp = axes[1, 0]
    ax_pier = axes[1, 1]
    ax_hanin = axes[2, 1]
    ax_humayan = axes[2, 0]
    ax_gradients = axes[0, 2]
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_sharp.set_title("Sharpness")
    ax_sharp.set_xlabel("epoch")
    ax_pier.set_title("Pier Regions")
    ax_pier.set_xlabel("epoch")
    ax_hanin.set_title("Hanin Regions")
    ax_hanin.set_xlabel("epoch")
    ax_humayan.set_title("Humayan Regions")
    ax_humayan.set_xlabel("epoch")
    ax_gradients.set_title("Gradients")
    ax_gradients.set_xlabel("epoch")
    axes[1, 2].axis("off")
    axes[2, 2].axis("off")

    epochs = len(history["train_loss"])

    def init():
        return []

    def update(i):
        ax_loss.clear()
        ax_acc.clear()
        ax_sharp.clear()
        ax_pier.clear()
        ax_hanin.clear()
        ax_humayan.clear()
        ax_gradients.clear()
        ax_loss.set_title("Loss")
        ax_acc.set_title("Accuracy")
        ax_sharp.set_title("Sharpness")
        ax_pier.set_title("Pier Regions")
        ax_hanin.set_title("Hanin Regions")
        ax_humayan.set_title("Humayan Regions")
        ax_gradients.set_title("Gradients")

        x = np.arange(1, (i + 1) * eig_freq + 1)
        x_eigs = np.arange(1, i + 1)
        x_regions = np.arange(1, i + 1)
        ax_loss.plot(x, history["train_loss"]
                     [: (i + 1) * eig_freq], label="train")
        ax_loss.plot(x, history["test_loss"]
                     [: (i + 1) * eig_freq], label="test")
        ax_loss.legend()

        ax_acc.plot(x, history["train_acc"]
                    [: (i + 1) * eig_freq], label="train")
        ax_acc.plot(x, history["test_acc"][: (i + 1) * eig_freq], label="test")
        ax_acc.legend()

        ax_sharp.plot(x_eigs, history["eigs"][:i])
        ax_sharp.axhline(y=2 / lr, color="red", linestyle="--", label="2/eta")
        if opt == "adam":
            ax_sharp.axhline(y=38 / lr, color="r",
                             linestyle="--", linewidth=1, label="38/lr")
        else:
            ax_sharp.axhline(y=2 / lr, color="r",
                             linestyle="--", linewidth=1, label="2/lr")
        ax_sharp.legend()
        ax_sharp.grid(True)

        ax_pier.plot(x_regions, history["regions_pier"][:i, 0])
        ax_pier.fill_between(
            x_regions,
            history["regions_pier"][:i, 1],
            history["regions_pier"][:i, 2],
            alpha=0.3,
        )
        ax_pier.set_ylim(0, num_samples_line)
        ax_pier.grid(True)

        ax_hanin.plot(x_regions, history["regions_hanin"][:i, 0])
        ax_hanin.fill_between(
            x_regions,
            history["regions_hanin"][:i, 1],
            history["regions_hanin"][:i, 2],
            alpha=0.3,
        )
        ax_hanin.set_ylim(0, num_hanin_line_samples)
        ax_hanin.grid(True)

        ax_humayan.plot(x_regions, history["regions_humayan"][:i, 0])
        ax_humayan.fill_between(
            x_regions,
            history["regions_humayan"][:i, 1],
            history["regions_humayan"][:i, 2],
            alpha=0.3,
        )
        ax_humayan.set_ylim(0, num_humayan_orthonormal_vectors)
        ax_humayan.grid(True)

        ax_gradients.hist(history["gradients"][i], bins=100, edgecolor="black")
        ax_gradients.grid(True)

        fig.suptitle(f"Epoch {i + 1}/{epochs}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=int(
        epochs / eig_freq), init_func=init, blit=False)
    anim.save(path + ".mp4", fps=4, dpi=150)
    print("Saved animation to", path + ".mp4")
    plt.close(fig)
