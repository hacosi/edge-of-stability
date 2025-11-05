import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(
    directory, train_loss, test_loss, train_acc, test_acc, eigs, eig_freq
):
    """
    Replaces save_files_final. Generates plots for loss, accuracy, and sharpness over training steps.
    """

    steps = np.arange(len(train_loss))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Summary", fontsize=16)

    # --- Plot train/test loss ---
    ax = axes[0, 0]
    ax.plot(steps, train_loss.cpu(), label="Train Loss")
    ax.plot(steps, test_loss.cpu(), label="Test Loss", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # --- Plot train/test accuracy ---
    ax = axes[0, 1]
    ax.plot(steps, train_acc.cpu(), label="Train Accuracy")
    ax.plot(steps, test_acc.cpu(), label="Test Accuracy", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    # --- Plot sharpness (top eigenvalues) ---
    if eig_freq > 0 and len(eigs) > 0:
        eig_steps = np.arange(0, len(train_loss), eig_freq)[: len(eigs)]
        ax = axes[1, 0]
        ax.plot(eig_steps, eigs.cpu())
        ax.set_xlabel("Step")
        ax.set_ylabel("Eigenvalues (Sharpness)")
        ax.set_title("Top Hessian Eigenvalues")
        ax.grid(True)
    else:
        axes[1, 0].axis("off")

    # --- Hide the last empty subplot or use it for summary text ---
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
