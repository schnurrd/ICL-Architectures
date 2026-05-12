import random

import torch

from .prior import PriorDataLoader
from torch import nn
import numpy as np
import scipy.stats as stats
import math

default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

def set_locals_in_self(locals):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals['self']
    for var_name, val in locals.items():
        if var_name != 'self': setattr(self, var_name, val)

def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num

def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))

def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)

def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)

def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data

def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
            # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X

def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features)**(1 / 2)
    return x / (num_features_used / num_features)

def to_ranking_low_mem(data: torch.Tensor) -> torch.Tensor:
    N, B, F = data.shape
    device = data.device
    out_dtype = data.dtype

    # (N, B, F) -> (rows=B*F, N)
    x = data.permute(1, 2, 0).reshape(-1, N)

    # sort ascending
    sv, sidx = torch.sort(x, dim=-1)

    if torch.is_floating_point(sv):
        valid = ~torch.isnan(sv)
    else:
        valid = None

    # group starts (only within valid region)
    ends = torch.zeros_like(sv, dtype=torch.bool)

    if valid is None:
        ends[:, :-1] = sv[:, :-1] != sv[:, 1:]
        ends[:, -1] = True
    else:
        both_valid = valid[:, :-1] & valid[:, 1:]
        ends[:, :-1] = both_valid & (sv[:, :-1] != sv[:, 1:])

        last_valid = valid.long().sum(dim=1) - 1
        has_any = last_valid >= 0
        ends[has_any, last_valid[has_any]] = True

    idx = torch.arange(N, device=device, dtype=torch.int32).unsqueeze(0).expand(sv.shape[0], -1)
    big = torch.full_like(idx, N, dtype=torch.int32)
    end_marker = torch.where(ends, idx, big)
    end_pos = torch.flip(torch.cummin(torch.flip(end_marker, dims=[1]), dim=1).values, dims=[1])

    ranks_sorted = (end_pos + 1).to(out_dtype)
    if valid is not None:
        ranks_sorted = torch.where(valid, ranks_sorted, torch.zeros(1, device=device, dtype=out_dtype))

    # scatter back to original order
    ranks_flat = torch.empty_like(x, dtype=out_dtype)
    ranks_flat.scatter_(1, sidx, ranks_sorted)

    # (rows, N) -> (N, B, F)
    return ranks_flat.reshape(B, F, N).permute(2, 0, 1)

def get_nan_value(v, set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])
    
def nan_handling_missing_for_unknown_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('nan'), set_value_to_nan)

def nan_handling_missing_for_no_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('-inf'), set_value_to_nan)

def nan_handling_missing_for_a_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float('inf'), set_value_to_nan)

def get_batch_to_dataloader(get_batch_method_):
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps, **get_batch_kwargs):
            set_locals_in_self(locals())

            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features = get_batch_kwargs.get('num_features') or self.num_features
            self.epoch_count = 0
            #print('DataLoader.__dict__', self.__dict__)

        @staticmethod
        def gbm(*args, eval_pos_seq_len_sampler, **kwargs):
            kwargs['single_eval_pos'], kwargs['seq_len'] = eval_pos_seq_len_sampler()
            # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
            # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
            if 'dynamic_batch_size' in kwargs and kwargs['dynamic_batch_size'] > 0 and kwargs['dynamic_batch_size']:
                kwargs['batch_size'] = kwargs['batch_size'] * math.floor(math.pow(kwargs['seq_len_maximum'], kwargs['dynamic_batch_size']) / math.pow(kwargs['seq_len'], kwargs['dynamic_batch_size']))
            batch = get_batch_method_(*args, **kwargs)
            x, y, target_y, style = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
            return (style, x, y), target_y, kwargs['single_eval_pos']

        def __len__(self):
            return self.num_steps

        def get_test_batch(self): # does not increase epoch_count
            return self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count, model=self.model if hasattr(self, 'model') else None)

        def __iter__(self):
            assert hasattr(self, 'model'), "Please assign model with `dl.model = ...` before training."
            self.epoch_count += 1
            return iter(self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1, model=self.model) for _ in range(self.num_steps))

    return DL

def plot_features(data, targets, fig=None, categorical=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    #data = np.concatenate([data, np.expand_dims(targets, -1)], -1)
    #df = pd.DataFrame(data, columns=list(range(0, data.shape[1])))
    #g = sns.pairplot(df, hue=data.shape[1]-1, palette="Set2", diag_kind="kde", height=2.5)
    #plt.legend([], [], frameon=False)
    #g._legend.remove()
    #g = sns.PairGrid(df, hue=data.shape[1]-1)
    #g.map_diag(sns.histplot)
    #g.map_offdiag(sns.scatterplot)
    #g._legend.remove()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if d == d2:
                if categorical:
                    sns.kdeplot(data[:, d],hue=targets[:],ax=sub_ax,legend=False, palette="deep")
                else:
                    sns.kdeplot(data[:, d], ax=sub_ax, legend=False)
                sub_ax.set(ylabel=None)
            else:
                if categorical:
                    sns.scatterplot(x=data[:, d], y=data[:, d2],
                           hue=targets[:],legend=False, palette="deep")
                else:
                    sns.scatterplot(x=data[:, d], y=data[:, d2],
                                    hue=targets[:], legend=False)
                #plt.scatter(data[:, d], data[:, d2],
                #               c=targets[:])
            #sub_ax.get_xaxis().set_ticks([])
            #sub_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig2.show()


def plot_prior(prior):
    import matplotlib.pyplot as plt
    s = np.array([prior() for _ in range(0, 1000)])
    count, bins, ignored = plt.hist(s, 50, density=True)
    print(s.min())
    plt.show()

trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : round(np.random.uniform(a, b))
def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda : stats.rv_discrete(name='bounded_zipf', values=(x, weights)).rvs(1)
scaled_beta_sampler_f = lambda a, b, scale, minimum : lambda : minimum + round(beta_sampler_f(a, b)() * (scale - minimum))

def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)#.reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y

def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7
                 , keep_activation_size=False
                 , num_classes_sampler=zipf_sampler_f(0.8, 1, 10)):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1],x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x