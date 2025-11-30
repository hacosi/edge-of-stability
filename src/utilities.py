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


def get_gd_path(
    dataset: str,
    lr: float,
    arch_id: str,
    seed: int,
    opt: str,
    loss: str,
    beta: float = None,
    beta1: float = None,
    beta2: float = None,
    eps: float = None,
):
    """Return the name for which the results png should be trained under."""
    path = f"{RESULTS_DIR}_{dataset}_{arch_id}_seed_{seed}_{loss}_{opt}"
    if opt == "gd":
        return f"{path}_lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{path}_lr_{lr}_beta_{beta}"
    elif opt == "adam":
        return f"{path}_lr_{lr}_beta1_{beta1}_beta2_{beta2}_eps_{eps}"


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


def get_adam_nu(optimizer) -> torch.Tensor:
    vec = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            vec.append(state["exp_avg_sq"].view(-1))
    return torch.cat(vec)


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


def _collect_preacts_for_batch_on_device(model: nn.Module, batch: torch.Tensor, device: Optional[str] = None):
    """
    Run forward pass for `batch` and collect pre-activations for each nn.ReLU module.
    All collected tensors are on the same device as the model (not moved to CPU here).
    Returns list of pre-activation tensors (each shape (N_batch, hidden_dim)).
    """

    def _make_hook():
        def _hook(mod, inp, out):
            # inp[0] is pre-activation for nn.ReLU; keep on the same device, detached.
            z = inp[0].detach()
            preacts.append(z)

        return _hook

    preacts: List[torch.Tensor] = []
    handles = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(_make_hook()))

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


def count_linear_regions(model, batch, device):
    preacts = _collect_preacts_for_batch_on_device(model, batch, device=device)

    if not preacts:
        return 1.0

    # Build binary masks on device and concatenate along feature axis
    masks = [(z > 0).to(torch.int8) for z in preacts]  # each (N_total, hidden)
    mask_concat = torch.cat(masks, dim=1)  # (N_total, total_hidden) on device

    # Move concatenated mask once to CPU for unique computations
    mask_concat_cpu = mask_concat.cpu()

    return torch.unique(mask_concat_cpu, dim=0).shape[0]


def num_linear_regions_pier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: Optional[str] = "cuda",
    num_samples_pairs: int = 10,
    num_samples_line: int = 10,
    max_attempts: int = 1000,
) -> Tuple[float, float, float]:
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
        if not idx1 == idx2 and not torch.equal(y[idx1], y[idx2]):
            x1 = X[idx1].unsqueeze(0).to(device)
            x2 = X[idx2].unsqueeze(0).to(device)
            alpha = torch.linspace(
                0, 1.0, steps=num_samples_line, device=device)
            alpha = alpha.view(-1, 1, 1, 1)
            pts = (1 - alpha) * x1 + alpha * x2
            lines_on_device.append(pts)
            accepted += 1

    if len(lines_on_device) == 0:
        return 1.0

    counts = []
    for batch in lines_on_device:
        counts.append(count_linear_regions(
            model=model, batch=batch, device=device))

    mean = np.mean(counts)
    std = np.std(counts)
    return mean, mean - std, mean + std


def num_linear_regions_hanin(
    model: nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_hanin_point_samples: int = 10,
    num_hanin_line_samples: int = 10,
    max_attempts: int = 1000,
) -> float:
    """
    Hanin-style but boundary-to-boundary:
    - For each sampled x from X, compute r_max = max_x ||x|| across X (data envelope).
    - Stretch the ray through x so endpoints are at +/- (r_max / ||x||) * x, i.e.
      the line segment crosses the data envelope in both directions (opposite endpoints).
    - Sample num_hanin_line_samples points along that segment (boundary-to-boundary).
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
            r_max = 1.0

    lines_on_device = []
    attempts = 0

    while attempts < num_hanin_point_samples:
        attempts += 1
        idx = torch.randint(0, N, (1,)).item()
        x = X[idx].to(device)
        norm_x = float(x.norm().item())
        if norm_x == 0.0:
            # degenerate sample (zero vector) — skip or create a small random direction
            # here we skip to get a meaningful direction
            continue

        # scaling factor so that ||s * xp|| = r_max  => s = r_max / ||xp||
        s = r_max / norm_x
        # endpoints are -s*xp and +s*xp (opposite directions through origin)
        e1 = -s * x
        e2 = +s * x
        a = torch.linspace(0.0, 1.0, steps=num_hanin_line_samples,
                           device=device).view(-1, 1, 1, 1)
        pts = (1 - a) * e1.unsqueeze(0) + a * e2.unsqueeze(0)
        lines_on_device.append(pts)

    if len(lines_on_device) == 0:
        return 1.0

    counts = []
    for batch in lines_on_device:
        counts.append(count_linear_regions(
            model=model, batch=batch, device=device))

    mean = np.mean(counts)
    std = np.std(counts)
    return mean, mean - std, mean + std


def num_linear_regions_humayan(
    model: nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_humayan_samples: int = 100,
    p: int = 10,
    scale: float = 1,
) -> float:
    # Sample point in the training or test set
    # Sample P orthonormal vectors in input space
    # Get convex hull neighborbood about the point
    # Take the vertices of convex hull and ?count linear regions on each?
    N = X.size(0)
    device = device or (
        next(model.parameters()).device if any(
            w.requires_grad for w in model.parameters()) else torch.device("cpu")
    )

    # compute data envelope radius (L2)
    with torch.no_grad():
        norms = X.to(device).norm(dim=1)
        r_max = float(norms.max().item()) if norms.numel() > 0 else 0.0
        if r_max == 0.0:
            r_max = 1.0

    attempts = 0
    points_on_device = []
    # Sample orthonormal vectors
    dtype = X.dtype
    N, d1, d2, d3 = X.shape
    D = d1 * d2 * d3
    if p > D:
        raise ValueError(f"p must be <= D (got p={p}, D={D})")
    X_flat = X.reshape(N, D).to(device=device, dtype=dtype)
    A = torch.randn((D, p), device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    diag_sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    Q = Q * scale

    while attempts < num_humayan_samples:
        attempts += 1
        idx = torch.randint(0, N, (1,)).item()
        x_flat = X_flat[idx]
        hull_flat = torch.cat(
            [x_flat.unsqueeze(1) + Q, x_flat.unsqueeze(1) - Q], dim=1)
        hull = hull_flat.T.view(2 * p, d1, d2, d3)
        points_on_device.append(hull)

    if len(points_on_device) == 0:
        return 1.0

    counts = []
    for batch in points_on_device:
        counts.append(count_linear_regions(
            model=model, batch=batch, device=device))

    mean = np.mean(counts)
    std = np.std(counts)
    return mean, mean - std, mean + std


def num_linear_regions_perturb(
    X,
    model,
    D,
    k,
    device: Optional[str] = "cuda",
):
    # Sample D points from X
    # Produce k random small pertubations, gather points then compute linear regions
    print(X.shape)
    N = X.size(0)
    device = device or (
        next(model.parameters()).device if any(
            w.requires_grad for w in model.parameters()) else torch.device("cpu")
    )

    for _ in range(D):
        idx = torch.randint(0, N, (1,)).item()
        x = X[idx].to(device)
        norm_x = float(x.norm().item())
        if norm_x == 0.0:
            # degenerate sample (zero vector) — skip or create a small random direction
            # here we skip to get a meaningful direction
            continue

    pass


def get_gradients(model):
    weights = []
    for param in model.parameters():
        weights.append(param.flatten())
    weights = torch.cat(weights)
    return weights.detach().cpu()


def tensor_to_jsonable(obj):
    """Convert tensors (and nested tensors) to Python lists."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_jsonable(x) for x in obj]
    return obj  # leave primitives unchanged


def get_grid_sampled_plane_regions(model, X, y, grid_samples):
    pass
