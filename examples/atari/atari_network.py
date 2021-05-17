import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional, Sequence, Type
from tianshou.utils.net.common import MLP

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(
                self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape)))
            self.output_dim = np.prod(action_shape)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state

class LfiwDQN(DQN):

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu"
    ):
        super().__init__(c, h, w, action_shape, device, features_only=True)
        self.dqn_net = nn.Sequential(
            nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, np.prod(action_shape)))
        # on-policy discriminator network
        self.opd_net = nn.Sequential(
            nn.Linear(self.output_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, np.prod(action_shape) + 1))
        self.output_dim = np.prod(action_shape)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
        # whether to output q value
        output_dqn = True,
        # whether to output the on-policiness of actions given a state
        output_opd = False,
    ) -> Tuple[torch.Tensor, Any]:
        features, state = super().forward(x, state, info)
        if output_dqn:
            q_values = self.dqn_net(features)
            return q_values, state
        if output_opd:
            opd = self.opd_net(features)
            return opd, state
        return features, state


ModuleType = Type[nn.Module]
class RamDQN(nn.Module):
    
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.device = device
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))

        self.net = MLP(input_dim, action_dim, hidden_sizes, 
                       norm_layer=None, activation=activation, 
                       device=device)
        self.output_dim = self.net.output_dim

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device)
        if len(x.shape) == 3 and x.shape[1] ==4:
            # remove stacked obs
            x = x[:, -1, :]
        return self.net(x), state

class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_atoms], device)
        self.num_atoms = num_atoms

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        x, state = super().forward(x)
        x = x.view(-1, self.num_atoms).softmax(dim=-1)
        x = x.view(-1, self.action_num, self.num_atoms)
        return x, state


class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_quantiles: int = 200,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_quantiles], device)
        self.num_quantiles = num_quantiles

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        x, state = super().forward(x)
        x = x.view(-1, self.action_num, self.num_quantiles)
        return x, state
