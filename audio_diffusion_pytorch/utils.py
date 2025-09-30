from functools import reduce
from inspect import isfunction
from math import ceil, floor, log2, pi
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Generator, Tensor
from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


"""
Kwargs Utils
"""


def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}


"""
DSP Utils
"""


def resample(
    waveforms: Tensor,
    factor_in: int,
    factor_out: int,
    rolloff: float = 0.99,
    lowpass_filter_width: int = 6,
) -> Tensor:
    """Resamples a waveform using sinc interpolation, adapted from torchaudio"""
    b, _, length = waveforms.shape
    length_target = int(factor_out * length / factor_in)
    d = dict(device=waveforms.device, dtype=waveforms.dtype)

    base_factor = min(factor_in, factor_out) * rolloff
    width = ceil(lowpass_filter_width * factor_in / base_factor)
    idx = torch.arange(-width, width + factor_in, **d)[None, None] / factor_in  # type: ignore # noqa
    t = torch.arange(0, -factor_out, step=-1, **d)[:, None, None] / factor_out + idx  # type: ignore # noqa
    t = (t * base_factor).clamp(-lowpass_filter_width, lowpass_filter_width) * pi

    window = torch.cos(t / lowpass_filter_width / 2) ** 2
    scale = base_factor / factor_in
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    waveforms = rearrange(waveforms, "b c t -> (b c) t")
    waveforms = F.pad(waveforms, (width, width + factor_in))
    resampled = F.conv1d(waveforms[:, None], kernels, stride=factor_in)
    resampled = rearrange(resampled, "(b c) k l -> b c (l k)", b=b)
    return resampled[..., :length_target]


def downsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    return resample(waveforms, factor_in=factor, factor_out=1, **kwargs)


def upsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    return resample(waveforms, factor_in=1, factor_out=factor, **kwargs)


""" Torch Utils """


def randn_like(tensor: Tensor, *args, generator: Optional[Generator] = None, **kwargs):
    """randn_like that supports generator"""
    return torch.randn(tensor.shape, *args, generator=generator, **kwargs).to(tensor)

# LoRA utilities for lightweight fine-tuning on Conv1d layers
from torch import nn  # placed after initial imports to avoid reordering large header


class LoRAConv1d(nn.Module):
    """
    Parallel LoRA adapter for Conv1d: y = Conv(x) + scale * B(A(x))
    - A (lora_down): Conv1d matching base temporal params to preserve length
    - B (lora_up): 1x1 Conv1d to mix back to out_channels
    The base conv is frozen; only A and B are trainable.
    """

    def __init__(self, base_conv: nn.Conv1d, rank: int = 8, alpha: int = 8):
        super().__init__()
        assert isinstance(base_conv, nn.Conv1d), "LoRAConv1d expects nn.Conv1d"
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        self.rank = max(int(rank), 1)
        self.scale = float(alpha) / float(self.rank)

        padding_mode = getattr(base_conv, 'padding_mode', 'zeros')
        # Down path matches temporal behavior of base conv to keep same length
        self.lora_down = nn.Conv1d(
            in_channels,
            self.rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            bias=False,
            padding_mode=padding_mode,
        )
        # Up path is 1x1 to mix channels only (no temporal change)
        self.lora_up = nn.Conv1d(self.rank, out_channels, kernel_size=1, bias=False)

        # Initialization: down ~ small normal, up = zeros so initial delta ~ 0
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        y = y + self.scale * self.lora_up(self.lora_down(x))
        return y


def inject_lora(module: nn.Module, rank: int = 8, alpha: int = 8, predicate: Optional[Callable[[str, nn.Module], bool]] = None) -> list:
    """
    Recursively replace eligible nn.Conv1d layers with LoRAConv1d.
    - predicate(name, submodule) -> bool can limit which layers are adapted.
    - Returns list of LoRA parameter tensors for optimizer grouping.
    """
    lora_params: list = []

    def default_pred(name: str, m: nn.Module) -> bool:
        # Adapt only 1D convs with kernel_size >= 3 and groups == 1
        if isinstance(m, nn.Conv1d):
            k = m.kernel_size[0]
            return (k >= 3) and (m.groups == 1)
        return False

    use_pred = predicate or default_pred

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d) and use_pred(name, child):
            wrapped = LoRAConv1d(child, rank=rank, alpha=alpha)
            # Move LoRA params to match base conv device/dtype
            device = child.weight.device
            dtype = child.weight.dtype
            wrapped = wrapped.to(device=device, dtype=dtype)
            setattr(module, name, wrapped)
            # collect LoRA params
            lora_params.extend(list(wrapped.lora_down.parameters()))
            lora_params.extend(list(wrapped.lora_up.parameters()))
        else:
            lora_params.extend(inject_lora(child, rank=rank, alpha=alpha, predicate=use_pred))

    return lora_params
