import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
import torch.nn.functional as F

class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())  # type: ignore

    def _target_q(self, buffer: ReplayBuffer, indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        target_q = target_q[np.arange(len(result.act)), result.act]
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = model(obs_, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        weight = to_torch_as(weight, q)
        td = r - q
        loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

class TPDQNPolicy(DQNPolicy):
    
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        bk_step: bool = False,
        reweigh_type: str = "hard",
        reweigh_hyper = None,
        env_fn = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model = model,
            optim = optim,
            discount_factor = discount_factor,
            estimation_step = estimation_step,
            target_update_freq = target_update_freq,
            reward_normalization = reward_normalization,
            **kwargs, 
        )
        self.tper_weight = reweigh_hyper["hard_weight"]
        self.bk_step = bk_step
        assert self.tper_weight <= 1.0
        self.reweigh_type = reweigh_type
        self.reweigh_hyper = reweigh_hyper
        if "linear" in self.reweigh_type:
            self.l, self.h, self.k, self.b = self.reweigh_hyper["linear"]
        if self.reweigh_type in ["adaptive_linear", "done_cnt_linear"]:
            self.low_l, self.low_h, self.high_l, self.high_h, self.t_s, self.t_e = \
                self.reweigh_hyper["adaptive_linear"]
        
        self.env_fn = env_fn

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        batch = super().process_fn(batch, buffer, indice)
        step = batch.step
        done_cnt = batch.done_cnt
        rel_step = step / np.max(step) if step.any() else np.zeros_like(step)
        if self.bk_step:
            # convert bk step to forward 
            rel_step = 1 - rel_step
        if self.reweigh_type == "hard":
            med = np.median(step)
            cond = step > med if self.bk_step else step < med
            weight = np.where(cond, self.tper_weight, 2 - self.tper_weight)
        elif self.reweigh_type == "linear":
            weight = self._calc_linear_weight(rel_step, self.l, self.h, self.k, self.b)
        elif self.reweigh_type == 'adaptive_linear':
            cur_low = np.clip(
                self.low_l + (self.low_h - self.low_l)/(self.t_e - self.t_s)*(self._iter - self.t_s), 
                self.low_l, 
                self.low_h
            )
            cur_high = np.clip(
                self.high_h + (self.high_l - self.high_h)/(self.t_e - self.t_s)*(self._iter - self.t_s), 
                self.high_l, 
                self.high_h
            )
            weight = self._calc_linear_weight(rel_step, cur_low, cur_high, self.k, self.b)
        elif self.reweigh_type == 'done_cnt_linear':
            rel_done_cnt = done_cnt / np.max(done_cnt)
            # The tajectory is newer with larger done counts, which can be understood as fewer learning steps
            pseudo_step = 1 - rel_done_cnt
            cur_low = np.clip(
                self.low_l + (self.low_h - self.low_l) * pseudo_step,
                self.low_l, 
                self.low_h
            )
            cur_high = np.clip(
                self.high_h + (self.high_l - self.high_h) * pseudo_step,
                self.high_l, 
                self.high_h
            )
            weight = self._calc_linear_weight(rel_step, cur_low, cur_high, self.k, self.b)
        elif self.reweigh_type == 'oracle':
            info = batch.info
            # assert "agent_pos" in info.keys()
            agent_pos = info["agent_pos"]
            reward = batch.rew
            done = batch.done
            action = batch.act
            # print(obs_next, reward, done)

            next_agent_pos = self._get_next_agent_pos(agent_pos, action)
            next_V = self._get_oracle_V(next_agent_pos)
            Qstar = reward + (1-done) * next_V
            with torch.no_grad():
                Qs = self.forward(batch).logits.detach().cpu().numpy()
            Qk = []
            for Q, a in zip(Qs, action):
                Qk.append(Q[a])
            weight = np.exp(-np.abs(Qk-Qstar))
            assert weight.shape[0] == rel_step.shape[0]
            weight = weight / np.sum(weight) * rel_step.shape[0]
        batch.update({"weight": weight})
        return batch

    def _get_oracle_V(self, next_agent_pos):
        oracle_V = []
        for pos in next_agent_pos:
            man_dist = 14 - pos[0] - pos[1]
            V = 0.99 ** man_dist
            oracle_V.append(V)
        return oracle_V
    def _get_next_agent_pos(self, agent_pos, action):
        next_agent_pos = []
        for (pos, act) in zip(agent_pos, action):
            env = self.env_fn(pos)
            env.reset()
            env.step(act)
            next_agent_pos.append(deepcopy(env.agent_pos))
        return next_agent_pos

    def _calc_linear_weight(self, rel_step, l, h, k, b):
        assert np.max(rel_step) <= 1
        assert np.min(rel_step) >= 0
        weight = np.clip(k*rel_step+b, l, h)
        weight = weight / np.sum(weight) * rel_step.shape[0]

        return weight

class LfiwTPDQNPolicy(TPDQNPolicy):

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        bk_step: bool = False,
        reweigh_type: str = "hard",
        reweigh_hyper = None,
        env_fn = None,
        opd_loss_coeff = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model = model,
            optim = optim,
            discount_factor = discount_factor,
            estimation_step = estimation_step,
            target_update_freq = target_update_freq,
            reward_normalization = reward_normalization,
            bk_step = bk_step,
            reweigh_type = reweigh_type,
            reweigh_hyper = reweigh_hyper,
            env_fn = env_fn,
            **kwargs,
        )
        self.opd_loss_coeff = opd_loss_coeff

    def forward(self, 
                batch: Batch, 
                state: Optional[Union[dict, Batch, np.ndarray]] = None, 
                model: str = "model", 
                input: str = "obs", 
                output_dqn = True,
                output_opd = False,
                **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = model(obs_, state=state, info=batch.info, 
                          output_dqn = output_dqn, 
                          output_opd = output_opd)
        if output_dqn:
            q = self.compute_q_value(logits, getattr(obs, "mask", None))
            if not hasattr(self, "max_action_num"):
                self.max_action_num = q.shape[1]
            act = to_numpy(q.max(dim=1)[1])
        else:
            act = None
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()

        # compute dqn loss
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        weight = to_torch_as(weight, q)
        td = r - q
        dqn_loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer

        # compute opd loss
        slow_preds = self(batch, output_dqn=False, output_opd=True).logits
        fast_preds = self(batch, input="fast_obs", output_dqn=False, output_opd=True).logits
        # act_dim + 1 classes. The last class indicate the state is off-policy
        slow_label = torch.ones_like(slow_preds[:, 0], dtype=torch.int64) * self.model.output_dim
        fast_label = to_torch_as(torch.tensor(batch.fast_act), slow_label)
        opd_loss = F.cross_entropy(slow_preds, slow_label) + \
                    F.cross_entropy(fast_preds, fast_label)

        # compute loss and back prop
        loss = dqn_loss + self.opd_loss_coeff * opd_loss
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": loss.item(), 
            "dqn_loss": dqn_loss.item(), 
            "opd_loss": opd_loss.item(),
        }

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        batch = super().process_fn(batch, buffer, indice)
        return batch