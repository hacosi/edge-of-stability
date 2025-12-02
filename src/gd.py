import os
import json
from os import makedirs

import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam, lr_scheduler

import argparse
from typing import Union

from archs import load_architecture
from utilities import (
    get_gd_optimizer,
    get_gd_directory,
    get_loss_and_acc,
    compute_losses,
    save_files,
    save_files_final,
    get_hessian_eigenvalues,
    iterate_dataset,
    get_gd_path,
    num_linear_regions_pier,
    num_linear_regions_hanin,
    num_linear_regions_humayan,
    get_adam_nu,
    get_gradients,
    tensor_to_jsonable,
    get_grid_sampled_plane_regions,
    num_linear_regions_perturb,
    num_linear_regions_PCA,
    num_linear_regions_directional_probe,
    num_linear_regions_low_dim_grid_search,
)
from data import load_dataset, take_first, DATASETS
from viz import plot_training_results, make_live_animation


def main(
    dataset: str,
    arch_id: str,
    loss: str,
    opt: str,
    lr: float,
    max_steps: int,
    neigs: int = 0,
    physical_batch_size: int = -1,
    eig_freq: int = -1,
    iterate_freq: int = -1,
    regions_freq: int = -1,
    save_freq: int = -1,
    save_model: bool = False,
    beta: float = 0.0,
    nproj: int = 0,
    loss_goal: float = None,
    acc_goal: float = None,
    abridged_size: int = 5000,
    seed: int = 0,
    num_samples_pairs: int = 10,
    num_samples_line: int = 10,
    num_hanin_point_samples: int = 10,
    num_hanin_line_samples: int = 10,
    num_humayan_samples: int = 10,
    num_humayan_orthonormal_vectors: int = 10,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    title: str = "",
    lr_schedule_gamma: float = 1,
    lr_schedule_steps: int = 1000,
    make_video: int = False,
    grid_samples: int = 200,
):
    # directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    # path = get_gd_path(dataset, lr, arch_id, seed, opt,
    #                    loss, beta, beta1, beta2, adam_epsilon)
    path = "_".join(title.split(" "))
    # print(f"output directory: {directory}")

    # makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_str = loss
    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    # projectors = torch.randn(nproj, len(
    #     parameters_to_vector(network.parameters())))

    if opt == "adam":
        optimizer = Adam(network.parameters(), lr=lr,
                         betas=(beta1, beta2), eps=adam_epsilon)
    else:
        print("Opt initialized with lr: ", lr)
        optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)

    # scheduler = lr_scheduler.StepLR(
    #     optimizer, step_size=1000, gamma=lr_schedule_gamma)
    # iterates = torch.zeros(
    #     max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))

    if physical_batch_size == -1:
        physical_batch_size = len(train_dataset)

    history = {
        "train_loss": torch.zeros(max_steps),
        "test_loss": torch.zeros(max_steps),
        "train_acc": torch.zeros(max_steps),
        "test_acc": torch.zeros(max_steps),
        "eigs": torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0),
        # "regions_pier": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_hanin": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_humayan": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_humayan_scale_0.1": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_humayan_scale_0.5": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_humayan_scale_0.01": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_humayan_scale_10": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        # "regions_perturb": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        # "regions_directional_probe": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        "regions_low_dim_grid_search": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        # "regions_pca_sample": torch.zeros((max_steps // regions_freq if regions_freq >= 0 else 0), 3),
        # "gradients": (
        #     torch.zeros(
        #         (max_steps // regions_freq if regions_freq >= 0 else 0),
        #         sum(len(param.flatten()) for param in network.parameters()),
        #     )
        #     if make_video
        #     else None
        # ),
        # "grid_sampled_plane_regions": (
        #     torch.zeros(
        #         (max_steps // regions_freq if regions_freq >= 0 else 0), grid_samples**2)
        #     if make_video
        #     else None
        # ),
    }

    for step in range(0, max_steps):
        # if step == 2500:
        #     lr = 0.01
        #     optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)
        #
        history["train_loss"][step], history["train_acc"][step] = compute_losses(
            network, [loss_fn, acc_fn], train_dataset, physical_batch_size
        )
        history["test_loss"][step], history["test_acc"][step] = compute_losses(
            network, [loss_fn, acc_fn], test_dataset, physical_batch_size
        )

        if opt == "adam":
            if step > 0 and eig_freq != -1 and step % eig_freq == 0:
                nu = get_adam_nu(optimizer)
                P = (1 - beta1**step) * \
                    ((nu / (1 - beta2**step)).sqrt() + adam_epsilon)
                history["eigs"][step // eig_freq] = get_hessian_eigenvalues(
                    network, loss_fn, abridged_train, neigs=neigs, physical_batch_size=physical_batch_size, P=P
                )
                print("eigenvalues: ", history["eigs"][step // eig_freq, :])
        else:
            if eig_freq != -1 and step % eig_freq == 0:
                history["eigs"][step // eig_freq] = get_hessian_eigenvalues(
                    network,
                    loss_fn,
                    abridged_train,
                    neigs=neigs,
                    physical_batch_size=physical_batch_size,
                )
                print("eigenvalues: ", history["eigs"][step // eig_freq])

        if regions_freq != -1 and step % regions_freq == 0:
            print("epoch ", step)

            x = train_dataset.tensors[0]
            y = train_dataset.tensors[1]

            # history["regions_pier"][step // regions_freq, :] = torch.tensor(
            #     num_linear_regions_pier(
            #         model=network, x=x, y=y, num_samples_pairs=num_samples_pairs, num_samples_line=num_samples_line
            #     )
            # )
            # print("pier regions: ",
            #       history["regions_pier"][step // regions_freq])
            history["regions_hanin"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_hanin(
                    model=network,
                    x=x,
                    num_hanin_point_samples=num_hanin_point_samples,
                    num_hanin_line_samples=num_hanin_line_samples,
                )
            )
            print("hanin regions: ",
                  history["regions_hanin"][step // regions_freq])
            history["regions_humayan"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_humayan(
                    model=network, x=x, num_humayan_samples=num_humayan_samples, p=num_humayan_orthonormal_vectors
                )
            )
            history["regions_humayan_scale_0.01"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_humayan(
                    model=network,
                    x=x,
                    num_humayan_samples=num_humayan_samples,
                    p=num_humayan_orthonormal_vectors,
                    scale=0.01,
                )
            )
            history["regions_humayan_scale_0.1"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_humayan(
                    model=network,
                    x=x,
                    num_humayan_samples=num_humayan_samples,
                    p=num_humayan_orthonormal_vectors,
                    scale=0.1,
                )
            )
            history["regions_humayan_scale_10"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_humayan(
                    model=network,
                    x=x,
                    num_humayan_samples=num_humayan_samples,
                    p=num_humayan_orthonormal_vectors,
                    scale=10,
                )
            )
            history["regions_humayan_scale_0.5"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_humayan(
                    model=network,
                    x=x,
                    num_humayan_samples=num_humayan_samples,
                    p=num_humayan_orthonormal_vectors,
                    scale=0.5,
                )
            )
            print("humayan regions: ",
                  history["regions_humayan"][step // regions_freq])
            # history["regions_perturb"][step // regions_freq, :] = torch.tensor(
            #     num_linear_regions_perturb(
            #         X=X,
            #         model=network,
            #     )
            # )
            # history["regions_directional_probe"][step // regions_freq, :] = torch.tensor(
            #     num_linear_regions_directional_probe(model=network, X=X)
            # )
            history["regions_low_dim_grid_search"][step // regions_freq, :] = torch.tensor(
                num_linear_regions_low_dim_grid_search(
                    model=network,
                    X=x,
                )
            )
            # history["regions_pca_sample"][step // regions_freq, :] = torch.tensor(
            #     num_linear_regions_PCA(model=network, X=X)
            # )
            #
            # if make_video:
            #     history["gradients"][step // regions_freq,
            #                          :] = get_gradients(model=network)
            #     history["grid_sampled_plane_regions"][step // regions_freq, :] = get_grid_sampled_plane_regions(
            #         model=network, X=X, y=y
            #     )

        # if iterate_freq != -1 and step % iterate_freq == 0:
        #     iterates[step // iterate_freq, :] = projectors.mv(
        #         parameters_to_vector(network.parameters()).cpu().detach())
        #
        # if save_freq != -1 and step % save_freq == 0:
        #     save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
        #                            ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
        #                            ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])
        #
        # print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{
        # test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        # if (loss_goal is not None and train_loss[step] < loss_goal) or (
        #     acc_goal is not None and train_acc[step] > acc_goal
        # ):
        #     print("Hit goal")
        #     break

        for X, y in iterate_dataset(train_dataset, physical_batch_size):
            optimizer.zero_grad()
            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss.backward()
            optimizer.step()
        # scheduler.step()
    if title == "":
        title = f"{dataset} | {arch_id} | {loss_str} | {opt} | lr {lr}"

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print("Generating plot of results...")
    plot_training_results(
        history,
        title,
        path,
        eig_freq,
        regions_freq,
        num_samples_line,
        lr,
        num_hanin_line_samples,
        num_humayan_orthonormal_vectors,
        lr_schedule_gamma,
        lr_schedule_steps,
        opt,
    )

    # if make_video:
    #     print("Generating moving animation of results...")
    #     make_live_animation(
    #         history=history,
    #         opt=opt,
    #         lr=lr,
    #         eig_freq=eig_freq,
    #         regions_freq=regions_freq,
    #         num_samples_line=num_samples_line,
    #         num_hanin_line_samples=num_hanin_line_samples,
    #         num_humayan_orthonormal_vectors=num_humayan_orthonormal_vectors,
    #         path=path,
    #     )
    #
    # print("Dumping results...")
    # with open("path.json", "w") as f:
    #     history_json_serializable = tensor_to_jsonable(history)
    #     json.dump(history_json_serializable, f, indent=2)

    # if save_model:
    #     torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS,
                        help="which dataset to train")
    parser.add_argument("arch_id", type=str,
                        help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=[
                        "ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int,
                        help="the maximum number of gradient steps to train for")
    parser.add_argument(
        "--opt",
        type=str,
        choices=["gd", "polyak", "nesterov", "adam"],
        help="which optimization algorithm to use",
        default="gd",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="the random seed used when initializing the network weights",
        default=0,
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="momentum parameter (used if opt = polyak or nesterov)",
    )
    parser.add_argument(
        "--physical_batch_size",
        type=int,
        help="the maximum number of examples that we try to fit on the GPU at once",
        default=-1,
    )
    parser.add_argument(
        "--acc_goal",
        type=float,
        help="terminate training if the train accuracy ever crosses this value",
    )
    parser.add_argument(
        "--loss_goal",
        type=float,
        help="terminate training if the train loss ever crosses this value",
    )
    parser.add_argument("--neigs", type=int,
                        help="the number of top eigenvalues to compute")
    parser.add_argument(
        "--eig_freq",
        type=int,
        default=-1,
        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)",
    )
    parser.add_argument("--nproj", type=int, default=0,
                        help="the dimension of random projections")
    parser.add_argument(
        "--iterate_freq",
        type=int,
        default=-1,
        help="the frequency at which we save random projections of the iterates",
    )
    parser.add_argument(
        "--abridged_size",
        type=int,
        default=5000,
        help="when computing top Hessian eigenvalues, use an abridged dataset of this size",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=-1,
        help="the frequency at which we save resuls",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="if 'true', save model weights at end of training",
    )
    parser.add_argument(
        "--regions_freq",
        type=int,
        default=-1,
    )
    parser.add_argument("--num_samples_pairs", type=int, default=10)
    parser.add_argument("--num_samples_line", type=int, default=10)
    parser.add_argument("--num_hanin_line_samples", type=int, default=10)
    parser.add_argument("--num_hanin_point_samples", type=int, default=10)
    parser.add_argument("--num_humayan_samples", type=int, default=10)
    parser.add_argument("--num_humayan_orthonormal_vectors",
                        type=int, default=10)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--lr_schedule_gamma", type=float, default=1)
    parser.add_argument("--lr_schedule_steps", type=int, default=1000)
    parser.add_argument("--make_video", type=bool, default=False)

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        arch_id=args.arch_id,
        loss=args.loss,
        opt=args.opt,
        lr=args.lr,
        max_steps=args.max_steps,
        neigs=args.neigs,
        physical_batch_size=args.physical_batch_size,
        eig_freq=args.eig_freq,
        iterate_freq=args.iterate_freq,
        save_freq=args.save_freq,
        save_model=args.save_model,
        beta=args.beta,
        nproj=args.nproj,
        loss_goal=args.loss_goal,
        acc_goal=args.acc_goal,
        abridged_size=args.abridged_size,
        seed=args.seed,
        regions_freq=args.regions_freq,
        num_samples_line=args.num_samples_line,
        num_samples_pairs=args.num_samples_pairs,
        num_hanin_line_samples=args.num_hanin_line_samples,
        num_hanin_point_samples=args.num_hanin_point_samples,
        num_humayan_samples=args.num_humayan_samples,
        num_humayan_orthonormal_vectors=args.num_humayan_orthonormal_vectors,
        title=args.title,
        lr_schedule_gamma=args.lr_schedule_gamma,
        lr_schedule_steps=args.lr_schedule_steps,
        make_video=args.make_video,
    )
