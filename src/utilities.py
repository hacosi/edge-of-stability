from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_gd_path(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the name for which the results png should be trained under."""
    path = f"{RESULTS_DIR}_{dataset}_{arch_id}_seed_{seed}_{loss}_{opt}"
    if opt == "gd":
        return f"{path}_lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{path}_lr_{lr}_beta_{beta}"


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    directory = f"{RESULTS_DIR}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}/"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    return f"{RESULTS_DIR}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    return f"{RESULTS_DIR}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for arr_name, arr in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for arr_name, arr in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_X, batch_y in loader:
        yield batch_X.cuda(), batch_y.cuda()


def compute_losses(
    network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset, batch_size: int = DEFAULT_PHYS_BS
):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0.0 for l in range(L)]
    with torch.no_grad():
        for X, y in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction."""
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction="sum"), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(
    network: nn.Module,
    loss_fn: nn.Module,
    dataset: Dataset,
    vector: Tensor,
    physical_batch_size: int = DEFAULT_PHYS_BS,
    P: Tensor = None,
):
    """Compute a Hessian-vector product.

    If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device="cuda")
    vector = vector.cuda()
    if P is not None:
        vector = vector / P.cuda().sqrt()
    for X, y in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(
            loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(
            dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    if P is not None:
        hvp = hvp / P.cuda().sqrt()
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products)."""

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), torch.from_numpy(
        np.ascontiguousarray(np.flip(evecs, -1)).copy()
    ).float()


def get_hessian_eigenvalues(
    network: nn.Module, loss_fn: nn.Module, dataset: Dataset, neigs=6, physical_batch_size=1000, P=None
):
    """Compute the leading Hessian eigenvalues.

    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    hvp_delta = (
        lambda delta: compute_hvp(
            network, loss_fn, dataset, delta, physical_batch_size=physical_batch_size, P=P)
        .detach()
        .cpu()
    )
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


def compute_gradient(
    network: nn.Module, loss_fn: nn.Module, dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS
):
    """Compute the gradient of the loss function at the current network parameters."""
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device="cuda")
    for X, y in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(
            torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(
    network: nn.Module, loss_fn: nn.Module, dataset: Dataset, theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS
):
    """Compute the gradient of the loss function at arbitrary network parameters "theta"."""
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()


class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()


class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()


class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0


def _make_hook(preacts: List[torch.Tensor]):
    def _hook(mod, inp, out):
        # inp[0] is pre-activation for nn.ReLU; keep on the same device, detached.
        z = inp[0].detach()
        preacts.append(z)

    return _hook


def _collect_preacts_for_batch_on_device(model: nn.Module, batch: torch.Tensor, device: Optional[str] = None):
    """
    Run forward pass for `batch` and collect pre-activations for each nn.ReLU module.
    All collected tensors are on the same device as the model (not moved to CPU here).
    Returns list of pre-activation tensors (each shape (N_batch, hidden_dim)).
    """
    preacts: List[torch.Tensor] = []
    handles = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(_make_hook(preacts)))

    # Keep/restore training mode
    was_training = model.training
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    with torch.no_grad():
        _ = model(batch.to(device))

    # remove hooks and restore training state
    for h in handles:
        h.remove()
    model.train(was_training)

    return preacts


def num_linear_regions_pier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: Optional[str] = "cuda",
    num_samples_pairs: int = 10,
    num_samples_line: int = 10,
    max_attempts: int = 1000,
) -> float:
    """
    Pier-style: sample pairs with different labels; for each pair sample num_samples_line
    points along the segment between them. Batch all line points, run forward once,
    compute per-line unique activation patterns and return the average count.
    """
    N = X.size(0)
    device = device or (
        next(model.parameters()).device if any(
            p.requires_grad for p in model.parameters()) else torch.device("cpu")
    )

    lines_on_device = []  # list of tensors on `device`, each shape (L, D)
    attempts = 0
    accepted = 0

    while accepted < num_samples_pairs and attempts < max_attempts:
        attempts += 1
        idx1 = torch.randint(0, N, (1,)).item()
        idx2 = torch.randint(0, N, (1,)).item()
        if idx1 == idx2:
            continue
        # compare labels robustly
        if not torch.equal(y[idx1], y[idx2]):
            x1 = X[idx1].to(device)
            x2 = X[idx2].to(device)
            # a = torch.linspace(0.0, 1.0, steps=num_samples_line,
            #                    device=device).unsqueeze(1)  # (L,1)
            #
            a = torch.linspace(0.0, 1.0, steps=num_samples_line, device=device).view(
                num_samples_line, *([1] * x1.dim())
            )
            pts = (1 - a) * x1 + a * x2  # (L, D)
            lines_on_device.append(pts)
            accepted += 1

    if len(lines_on_device) == 0:
        return 1.0

    # Batch all line points into one big tensor on device
    batch = torch.cat(lines_on_device, dim=0)  # (num_lines * L, D) on device

    # Run forward pass and collect preacts (on device)
    preacts = _collect_preacts_for_batch_on_device(model, batch, device=device)

    breakpoint()
    if not preacts:
        return 1.0

    # Build binary masks on device and concatenate along feature axis
    masks = [(z > 0).to(torch.int8) for z in preacts]  # each (N_total, hidden)
    mask_concat = torch.cat(masks, dim=1)  # (N_total, total_hidden) on device

    # Move concatenated mask once to CPU for unique computations
    mask_concat_cpu = mask_concat.cpu()

    L = num_samples_line
    num_lines = len(lines_on_device)
    regions_per_line = []
    for i in range(num_lines):
        start = i * L
        end = start + L
        seg = mask_concat_cpu[start:end]  # (L, total_hidden), on CPU
        unique_patterns = torch.unique(seg, dim=0)
        regions_per_line.append(unique_patterns.shape[0])

    return float(np.mean(regions_per_line))


def num_linear_regions_hanin(
    model: nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_samples_pairs: int = 10,
    num_samples_line: int = 10,
    max_attempts: int = 1000,
) -> float:
    """
    Hanin-style but boundary-to-boundary:
    - For each sampled xp from X, compute r_max = max_x ||x|| across X (data envelope).
    - Stretch the ray through xp so endpoints are at +/- (r_max / ||xp||) * xp, i.e.
      the line segment crosses the data envelope in both directions (opposite endpoints).
    - Sample num_samples_line points along that segment (boundary-to-boundary).
    - Batch, forward once, compute unique activation patterns per line.
    Returns the average number of unique patterns along sampled lines.
    """
    N = X.size(0)
    device = device or (
        next(model.parameters()).device if any(
            p.requires_grad for p in model.parameters()) else torch.device("cpu")
    )

    # compute data envelope radius (L2)
    with torch.no_grad():
        norms = X.to(device).norm(dim=1)
        r_max = float(norms.max().item()) if norms.numel() > 0 else 0.0
        if r_max == 0.0:
            # all-zero dataset -> nothing to do; fall back to xp->-xp (or just zero)
            r_max = 1.0

    lines_on_device = []
    attempts = 0
    accepted = 0

    while accepted < num_samples_pairs and attempts < max_attempts:
        attempts += 1
        idxp = torch.randint(0, N, (1,)).item()
        xp = X[idxp].to(device)
        norm_xp = float(xp.norm().item())
        if norm_xp == 0.0:
            # degenerate sample (zero vector) — skip or create a small random direction
            # here we skip to get a meaningful direction
            continue

        # scaling factor so that ||s * xp|| = r_max  => s = r_max / ||xp||
        s = r_max / norm_xp
        # endpoints are -s*xp and +s*xp (opposite directions through origin)
        e1 = -s * xp
        e2 = +s * xp
        a = torch.linspace(0.0, 1.0, steps=num_samples_line,
                           device=device).unsqueeze(1)  # (L,1)
        pts = (1 - a) * e1.unsqueeze(0) + a * e2.unsqueeze(0)  # (L, D)
        lines_on_device.append(pts)
        accepted += 1

    if len(lines_on_device) == 0:
        return 1.0

    # Batch all line points into one big tensor on device
    batch = torch.cat(lines_on_device, dim=0)  # (num_lines * L, D) on device

    # Run forward pass and collect preacts (on device)
    preacts = _collect_preacts_for_batch_on_device(model, batch, device=device)

    if not preacts:
        return 1.0

    # Build binary masks on device and concatenate along feature axis
    masks = [(z > 0).to(torch.int8) for z in preacts]
    mask_concat = torch.cat(masks, dim=1)  # (N_total, total_hidden) on device

    # Move concatenated mask once to CPU for unique computations
    mask_concat_cpu = mask_concat.cpu()

    L = num_samples_line
    num_lines = len(lines_on_device)
    regions_per_line = []
    for i in range(num_lines):
        start = i * L
        end = start + L
        seg = mask_concat_cpu[start:end]
        unique_patterns = torch.unique(seg, dim=0)
        regions_per_line.append(unique_patterns.shape[0])

    return float(np.mean(regions_per_line))


# def _make_hook():
#     def _hook(mod, inp, out):
#         z = inp[0].detach().cpu()
#         preacts.append(z)
#
#     return _hook
#
#
# def num_linear_regions_pier(
#     model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str = "cpu", num_samples_pairs: int = 10, num_samples_line: int = 10,
# ) -> int:
#     """
#     Computes the number of linear regions in the following way:
#     1. Sample two points with different labels in the input space
#     2. Sample points in the input space that lie on the line between the two different label points
#     3. Run a forward pass on these points
#     4. Count the number of linear regions along the line by checking the activation pattens
#     5. Repeat from step 1 and average the results
#
#     This computation is quite slow, so only perform it occasionally
#     """
#     num_regions_all = []
#     for _ in range(num_samples_pairs):
#         idx1 = torch.randint(0, X.size(0), (1,)).item()
#         idx2 = torch.randint(0, X.size(0), (1,)).item()
#         ys_on_line = []
#         if y[idx1] != y[idx2]:  # different labels
#             x1, x2 = X[idx1], X[idx2]
#             for a in np.linspace(0, 1, num_samples_line):
#                 x = x1 * (1 - a) + x2 * a
#                 x = x.unsqueeze(0)
#                 yh = model(x)
#                 ys_on_line.append(yh)
#             num_samples_pairs -= 1
#
#         num_regions = 0
#         for i, _ in enumerate(ys_on_line[1:-1]):
#             y_delta_2 = ys_on_line[i + 1] - ys_on_line[i]
#             y_delta_1 = ys_on_line[i] - ys_on_line[i - 1]
#             if torch.norm(y_delta_2 - y_delta_1) > 1e-5:
#                 num_regions += 1
#         num_regions_all.append(num_regions)
#     return np.mean(num_regions_all)
#
#
# def num_linear_regions_hanin(
#     model: nn.Module, X: torch.Tensor, device: str = "cpu"
# ) -> int:
#     """
#     Computes the number of linear regions the same as in the "Pier Method" but one of the two sampled points is always the origin
#     """
#     # Sample a point from the training data
#     # Compute the number of linear regions along that line?
#     # For 5 independent runs, sample 100 lines, take average
#     num_regions_all = []
#     for _ in range(num_samples_pairs):
#         idxp = torch.randint(0, X.size(0), (1,)).item()
#         ys_on_line = []
#         xp = X[idxp]
#         for a in np.linspace(0, 1, num_samples_line):
#             x = xp * (1 - a) + x2 * a
#             # should go from boundary to boundary
#             x = x.unsqueeze(0)
#             yh = model(x)
#             ys_on_line.append(yh)
#         num_samples_pairs -= 1
#
#
#         num_regions = 0
#         for i, _ in enumerate(ys_on_line[1:-1]):
#             y_delta_2 = ys_on_line[i + 1] - ys_on_line[i]
#             y_delta_1 = ys_on_line[i] - ys_on_line[i - 1]
#             if torch.norm(y_delta_2 - y_delta_1) > 1e-5:
#                 num_regions += 1
#         num_regions_all.append(num_regions)
#     return np.mean(num_regions_all)
#
#
# def COMPUTE LINEAR REGIONS OF BATCH
#     preacts: List[torch.Tensor] = []
#     handles = []
#
#     for m in model.modules():
#         if isinstance(m, nn.ReLU):
#             handles.append(m.register_forward_hook(_make_hook()))
#
#     _ = model(X.to(next(model.parameters()).device))
#
#     for h in handles:
#         h.remove()
#
#     if not preacts:
#         return 1
#
#     masks = [(z > 0).to(torch.int8) for z in preacts]  # (N, hidden)
#     mask_concat = torch.cat(masks, dim=1)  # (N, total_hidden)
#
#     unique_patterns = torch.unique(mask_concat, dim=0)
#     num_regions = unique_patterns.shape[0]
#
#     return num_regions
#
#
#
#
def num_linear_regions_humayan():
    # Sample point in the training or test set
    # Sample P orthonormal vectors in input space
    # Get convex hull neighborbood about the point
    # Take the vertices of convex hull and ?count linear regions on each?
    pass
