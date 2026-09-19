import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
# from torch_scatter import scatter
import torch_geometric as pyg
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    """Simple replacement for torch_scatter.scatter using PyTorch operations"""
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    # Create output tensor
    size = list(src.shape)
    if dim < 0:
        dim = len(size) + dim
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Use scatter_add for sum reduction
    if reduce == 'sum':
        out.scatter_add_(dim, index, src)
    else:
        raise NotImplementedError(f"Reduce operation '{reduce}' not implemented")
    
    return out


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sub_exp(a,b):
    assert torch.any(a > b), f'Error: {a > b}'
    return a + torch.log1p(-torch.exp(b-a))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def create_node_selections(log_x_t, log_x_tminus1, batched_graph):
    d_t = scatter(log_x_t.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_t.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes) 
    d_tminus1 = scatter(log_x_tminus1.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_tminus1.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes)
    return (d_tminus1 > d_t).long()


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return 1 - torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).numpy()


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def Tt1_beta_schedule(timesteps):
    return 1/torch.linspace(1+1e-8, timesteps+1e-8, timesteps, dtype = torch.float64).flip(0).numpy()


def minimum_density_schedule(timesteps, base_schedule, rho_min=0.1):
    """Constrained schedule implementing η_{ε_t} = min(1 - ∏(1-ε_i), 1 - ρ_min).

    Wraps any base schedule and clamps the cumulative keep-probability to ≥ rho_min,
    guaranteeing at least that fraction of edges survive at every diffusion step.
    Steps where the base schedule would push density below rho_min become no-ops
    (per-step alpha = 1) so the minimum connectivity bound is never violated.
    """
    alphas = base_schedule(timesteps)
    cumprod_alpha = np.cumprod(alphas)
    constrained_cumprod = np.maximum(cumprod_alpha, rho_min)
    constrained_alphas = np.empty_like(alphas)
    constrained_alphas[0] = constrained_cumprod[0]
    constrained_alphas[1:] = constrained_cumprod[1:] / constrained_cumprod[:-1]
    return np.clip(constrained_alphas, a_min=0.001, a_max=1.0)


class DiffusionBase(torch.nn.Module):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                sample_time_method='importance', device='cuda'):
        super(DiffusionBase, self).__init__()

        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

        self._denoise_fn = denoise_fn
        self.initial_graph_sampler = initial_graph_sampler
        self.num_timesteps = timesteps
        self.sample_time_method = sample_time_method
        self._device = device

    @property
    def device(self):
        # Dynamically resolve the device so DataParallel replicas on different
        # GPUs each return their own device rather than the init-time device.
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self._device)
        

    def forward(self, *args, mode='log_prob', **kwargs):
        if mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        raise ValueError(f'Unknown mode: {mode}')

    def _q_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _p_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _prepare_data_for_sampling(self, batched_graph):
        raise NotImplementedError()

    def _eval_loss(self, batched_graph):
        raise NotImplementedError()

    def _train_loss(self, batched_graph):
        raise NotImplementedError()

    def _sample_time(self, b, device, method):
        raise NotImplementedError()
    def _calc_num_entries(self, batched_graph):
        raise NotImplementedError()

    def multinomial_kl(self, log_prob1, log_prob2):
        # When p=0 (log_prob1=-inf) and q=0 (log_prob2=-inf), the term is 0 by convention
        # but naive computation gives 0 * (-inf - -inf) = 0 * NaN = NaN.
        kl = log_prob1.exp() * (log_prob1 - log_prob2)
        kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
        return kl.sum(dim=1)


    def q_sample(self, batched_graph, t_node, t_edge):

        log_prob_node, log_prob_edge = self._q_pred(batched_graph, t_node, t_edge)

        # sample nodes
        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge 
    
    @torch.no_grad()
    def p_sample(self, batched_graph, t_node, t_edge):
        # p_sample is always one step prediction!
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)
        
        log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)
        return log_out_node, log_out_edge

    def log_sample_categorical(self, logits, num_classes):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample


    def log_prob(self, batched_graph):
        if self.training:
            return self._train_loss(batched_graph)
        else:
            return self._eval_loss(batched_graph)


    def _build_discrete_skip_schedule(self, inference_steps=None):
        """Return list of (t, t_prev) pairs for the discrete reverse chain.

        t runs from T-1 down to 0 (0-indexed, matching the existing loop).
        With inference_steps=None → full T steps.
        With inference_steps=K < T → K uniformly-spaced steps covering [0,T-1].
        t_prev=-1 signals the final deterministic step.
        """
        T = self.num_timesteps
        K = inference_steps if inference_steps is not None else T

        if K >= T:
            seq = list(range(T - 1, -1, -1))           # T-1, T-2, ..., 0
        else:
            import numpy as np
            seq = sorted(
                set(np.round(np.linspace(0, T - 1, K)).astype(int)),
                reverse=True)

        pairs = [(seq[i], seq[i + 1] if i + 1 < len(seq) else -1)
                 for i in range(len(seq))]
        return pairs

    def sample(self, num_samples, inference_steps=None):
        """Generate graphs via the learned reverse-diffusion chain.

        inference_steps: None → full T steps.  K < T → K uniformly-spaced
        reverse steps using the correct q(x_{t_prev}|x_t, x_0) posterior.
        """
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        step_seq = self._build_discrete_skip_schedule(inference_steps)
        n_steps  = len(step_seq)

        print()
        for step_i, (t, t_prev) in enumerate(step_seq):
            print(f'Sample step {step_i+1:4d}/{n_steps}  (t={t})', end='\r')
            t_node = torch.full((num_nodes,), t,      device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t,      device=self.device, dtype=torch.long)

            # t_prev tensors — used by _q_posterior for skip schedule
            if t_prev >= 0:
                tp_node = torch.full((num_nodes,), t_prev, device=self.device, dtype=torch.long)
                tp_edge = torch.full((num_edges,), t_prev, device=self.device, dtype=torch.long)
            else:
                tp_node = tp_edge = None                                    # final step → t_prev=t-1

            log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.p_sample(
                batched_graph, t_node, t_edge,
                t_prev_node=tp_node, t_prev_edge=tp_edge)
            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            batched_graph.log_node_attr_t      = log_node_attr_tmin1

        print()
        edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        batched_graph.edge_index = edge_index 

        edge_attr = edge_attr[is_edge_indices]
        batched_graph.edge_attr = edge_attr

        
        batched_graph.node_attr = batched_graph.log_node_attr_t.argmax(-1)

        # preparation for splitting batched graph
        # see https://github.com/pyg-team/pytorch_geometric/blob/259cfa7fb220d9cb504ab9de52bcd9dc5267befe/torch_geometric/data/separate.py#L12
        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs )
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']
        # edge_attr (bond types for actual edges) must share the same slice as edge_index
        batched_graph._slice_dict['edge_attr'] = edge_slice
        batched_graph._inc_dict['edge_attr']   = torch.zeros(
            batched_graph.num_graphs + 1, dtype=torch.long,
            device=batched_graph.edge_attr.device)

        return batched_graph
