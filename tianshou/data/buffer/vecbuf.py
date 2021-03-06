import numpy as np
from typing import Any

from tianshou.data import ReplayBuffer, ReplayBufferManager, TPRBManager, TPRBDoubleManager
from tianshou.data import PrioritizedReplayBuffer, PrioritizedReplayBufferManager


class VectorReplayBuffer(ReplayBufferManager):
    """VectorReplayBuffer contains n ReplayBuffer with the same size.

    It is used for storing transition from different environments yet keeping the order
    of time.

    :param int total_size: the total size of VectorReplayBuffer.
    :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
        configuration.

    Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
    are the same as :class:`~tianshou.data.ReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)

class TPVectorReplayBuffer(TPRBManager):
    def __init__(self, total_size: int, buffer_num: int, bk_step: bool = False, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list, bk_step=bk_step)

class TPDoubleVectorReplayBuffer(TPRBDoubleManager):
    def __init__(self, total_size: int, buffer_num: int, bk_step: bool = False, 
                 fast_buffer_size: int = None, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list, bk_step=bk_step, fast_buffer_size=fast_buffer_size)
class PrioritizedVectorReplayBuffer(PrioritizedReplayBufferManager):
    """PrioritizedVectorReplayBuffer contains n PrioritizedReplayBuffer with same size.

    It is used for storing transition from different environments yet keeping the order
    of time.

    :param int total_size: the total size of PrioritizedVectorReplayBuffer.
    :param int buffer_num: the number of PrioritizedReplayBuffer it uses, which are
        under the same configuration.

    Other input arguments (alpha/beta/stack_num/ignore_obs_next/save_only_last_obs/
    sample_avail) are the same as :class:`~tianshou.data.PrioritizedReplayBuffer`.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [
            PrioritizedReplayBuffer(size, **kwargs) for _ in range(buffer_num)
        ]
        super().__init__(buffer_list)
