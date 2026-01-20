from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def points_per_regions(model, batch, device):
    preacts = _collect_preacts_for_batch_on_device(model, batch, device=device)

    if not preacts:
        # All points fall into a single (trivial) region
        counts = torch.tensor([batch.shape[0]], dtype=torch.long)
    else:
        # Build binary masks on device and concatenate along feature axis
        masks = [(z > 0).to(torch.int8) for z in preacts]  # each (N, hidden)
        mask_concat = torch.cat(masks, dim=1)  # (N, total_hidden)

        # Move once to CPU for unique computation
        mask_concat_cpu = mask_concat.cpu()

        # Counts = number of points in each region (one entry per unique region)
        _, counts = torch.unique(
            mask_concat_cpu, dim=0, return_counts=True)  # (num_regions,)

    # Build "region-size histogram":
    # size k -> how many regions have exactly k points
    # Example: counts = [1,1,2,5] => {1:2 regions, 2:1 region, 5:1 region}
    size_to_num_regions = torch.bincount(
        counts)  # index is "points per region"
    # size_to_num_regions[0] is always 0 here; ignore it.

    # Return as a dict like: {"1 pt per region": 12, "2 pts per region": 3, ...}
    out = {f"{k}" for k in range(
        1, size_to_num_regions.numel()) if size_to_num_regions[k] > 0}
    return out


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
        if not idx1 == idx2 and torch.equal(y[idx1], y[idx2]):
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


def _process_groups_and_stats(model: torch.nn.Module, groups: list, device: Optional[str] = "cuda"):
    """
    groups: list of torch.Tensor each shape (m,3,32,32)
    returns: mean, mean-std, mean+std
    """
    if len(groups) == 0:
        return 1.0, 1.0, 1.0

    device = device or (
        next(model.parameters()).device if any(
            p.requires_grad for p in model.parameters()) else torch.device("cpu")
    )

    counts = []
    for g in groups:
        # ensure float32 and on device
        batch = g.to(device).float()
        c = count_linear_regions(model=model, batch=batch, device=device)
        counts.append(float(c))
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    return mean, mean - std, mean + std


def num_linear_regions_perturb(
    X,
    model,
    num_anchors: int = 500,
    per_anchor_augment: int = 100,
    noise_sigma: float = 0.01,
    max_translate: int = 1,
    device: Optional[str] = "cuda",
):
    # print(X.shape) -> [5000, 3, 32, 32]
    N = X.size(0)
    rng = torch.Generator()
    rng.manual_seed(torch.seed() % (2**31 - 1))
    idxs = torch.randperm(N, generator=rng)[:num_anchors]

    groups = []
    for idx in idxs:
        anchor = X[idx]  # shape (3,32,32)
        perturbs = []
        for _ in range(per_anchor_augment):
            x = anchor.clone()
            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                # flip width axis (C,H,W) -> flip last dim
                x = torch.flip(x, dims=[2])

            # small translation: pad then random crop back to 32x32
            if max_translate > 0:
                pad = max_translate
                x_padded = F.pad(x.unsqueeze(
                    0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
                top = torch.randint(0, 2 * pad + 1, (1,)).item()
                left = torch.randint(0, 2 * pad + 1, (1,)).item()
                x = x_padded[:, top: top + 32, left: left + 32]

            # per-channel brightness jitter
            scale = torch.empty(3).uniform_(0.9, 1.1)
            x = (x * scale.view(3, 1, 1)).clamp(0.0, 1.0)

            # additive gaussian noise
            x = x + torch.randn_like(x) * noise_sigma
            x = x.clamp(0.0, 1.0)
            perturbs.append(x)

        # (per_anchor_augment,3,32,32)
        groups.append(torch.stack(perturbs, dim=0))

    return _process_groups_and_stats(model, groups, device=device)


def num_linear_regions_directional_probe(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_anchors: int = 500,
    n_dirs: int = 8,
    steps: int = 100,
    eps: float = 2,
) -> Tuple[float, float, float]:
    """
    For each of num_anchors anchors, sample n_dirs random directions (Gaussian),
    normalize them and sample 'steps' points along t in [-eps, eps].
    Each (direction,line) is one group of size `steps`.
    """
    N = X.size(0)
    rng = torch.Generator()
    rng.manual_seed(torch.seed() % (2**31 - 1))
    idxs = torch.randperm(N, generator=rng)[:num_anchors]
    D = int(X[0].numel())  # 3*32*32

    groups = []
    for idx in idxs:
        x0 = X[idx].reshape(-1)  # flatten
        if float(x0.norm().item()) == 0.0:
            continue
        x0_np = x0.cpu().numpy()
        for _ in range(n_dirs):
            v = torch.randn(D)
            v = v / (v.norm() + 1e-12)
            ts = torch.linspace(-eps, eps, steps)
            pts = x0.unsqueeze(0) + (ts.unsqueeze(1) @
                                     v.unsqueeze(0))  # (steps, D)
            pts = pts.view(steps, *X.shape[1:])  # .clamp(0.0, 1.0)
            groups.append(pts)

    return _process_groups_and_stats(model, groups, device=device)


def num_linear_regions_low_dim_grid_search(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_anchors: int = 500,
    d: int = 2,
    grid_size: int = 20,
    eps: float = 2,
) -> Tuple[float, float, float]:
    """
    For each anchor, create a random d-dimensional orthonormal basis U (D x d).
    Sample a grid in [-eps, eps]^d with grid_size points per axis (grid_size^d points).
    Each subspace (the full grid) is one group passed to count_linear_regions.
    Defaults: d=2, grid_size=20 -> 400 points per group.
    """
    N = X.size(0)
    D = int(X[0].numel())
    rng = torch.Generator()
    rng.manual_seed(torch.seed() % (2**31 - 1))
    idxs = torch.randperm(N, generator=rng)[:num_anchors]

    # prepare grid in parameter space
    if d == 1:
        # (grid_size,1)
        coords = torch.linspace(-eps, eps, grid_size).unsqueeze(1)
        Z = coords
    else:
        # create grid points using meshgrid (careful with dimensionality)
        axes = [torch.linspace(-eps, eps, grid_size) for _ in range(d)]
        mesh = torch.meshgrid(*axes, indexing="ij")
        Z = torch.stack([m.reshape(-1)
                        for m in mesh], dim=1)  # (grid_size^d, d)

    groups = []
    for idx in idxs:
        anchor = X[idx].reshape(-1).cpu().numpy()  # D
        # random gaussian then QR to make orthonormal basis
        G = np.random.randn(D, d)
        Q, _ = np.linalg.qr(G)
        U = torch.tensor(Q[:, :d].astype(np.float32))  # (D,d)
        # sample offsets
        Z_np = Z.numpy() if isinstance(Z, torch.Tensor) else Z
        # map to input space: anchor + Z @ U^T
        # Z (M,d), U^T (d,D) => (M, D)
        M = Z_np.shape[0]
        offsets = Z_np @ U.cpu().numpy().T  # (M,D)
        pts = torch.from_numpy(offsets.astype(np.float32)) + \
            torch.from_numpy(anchor.astype(np.float32))[None, :]
        pts = pts.view(M, *X.shape[1:]).clamp(0.0, 1.0)
        groups.append(pts)

    return _process_groups_and_stats(model, groups, device=device)


def num_linear_regions_PCA(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: Optional[str] = "cuda",
    num_pca_samples: int = 2000,
    num_components: int = 2,
    num_anchors: int = 500,
    grid_size: int = 20,
    eps: float = 2,
) -> Tuple[float, float, float]:
    """
    Grid search over the top `num_components` principal components.

    - Compute PCA on up-to `num_pca_samples` points from X.
    - For each anchor (num_anchors), build a regular grid in [-eps, eps]^num_components
      with `grid_size` points per axis (grid_size**num_components points).
    - Map grid points back into input space with anchor + Z @ PC_matrix.T and clamp to [0,1].
    - Each anchor's full grid is treated as one group for count_linear_regions.
    - Returns mean, mean-std, mean+std across anchors (groups).

    Defaults produce groups of size 20^2 = 400 when num_components=2.
    """
    N = X.size(0)
    device = device or (
        next(model.parameters()).device if any(
            p.requires_grad for p in model.parameters()) else torch.device("cpu")
    )

    # 1) PCA on a subset
    sample_N = min(int(N), int(num_pca_samples))
    rng_idxs = torch.randperm(N)[:sample_N]
    subset = X[rng_idxs].reshape(
        sample_N, -1).to(torch.float32)  # (sample_N, D)
    mean = subset.mean(dim=0, keepdim=True)
    centered = subset - mean  # (sample_N, D)

    # compute SVD to get principal directions (rows of Vh)
    try:
        # Use torch.linalg.svd; on GPU this is usually OK for modest sample_N
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        # fallback: move to cpu if necessary
        U, S, Vh = torch.linalg.svd(centered.cpu(), full_matrices=False)

    D = centered.shape[1]
    if num_components > D:
        raise ValueError(
            f"num_components ({num_components}) cannot exceed data dimension ({D}).")

    # (num_components, D)  <-- each row is a principal direction
    pcs = Vh[:num_components, :]

    # 2) prepare grid coords in PCA coefficient space
    axes = [torch.linspace(-eps, eps, grid_size)
            for _ in range(num_components)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    Z = torch.stack([m.reshape(-1)
                    for m in mesh], dim=1)  # (M, num_components)
    M = Z.shape[0]  # grid_size**num_components

    # 3) select anchors
    num_anchors = min(int(num_anchors), N)
    anchor_idxs = torch.randperm(N)[:num_anchors]

    groups = []
    # We'll convert pcs to numpy for faster mapping if needed
    pcs_np = pcs.cpu().numpy()  # (num_components, D)
    Z_np = Z.numpy()  # (M, num_components)

    for ai in anchor_idxs:
        anchor = X[ai].reshape(-1).cpu().numpy()  # (D,)
        # Map coefficients to input offsets: offsets = Z @ pcs_np  -> (M, D)
        offsets = Z_np @ pcs_np  # (M, D)
        pts = offsets + anchor[None, :]  # (M, D)
        pts_t = torch.from_numpy(pts.astype(np.float32)).reshape(
            M, *X.shape[1:]).clamp(0.0, 1.0)
        groups.append(pts_t)

    return _process_groups_and_stats(model, groups, device=device)


# def generator_sampling(X, model):
#     pass


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
