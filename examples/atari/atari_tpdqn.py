import os
from tianshou.data.batch import Batch
import time
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import TPDQNPolicy, LfiwTPDQNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, TPVectorReplayBuffer, TPDoubleVectorReplayBuffer

from atari_network import DQN, RamDQN, LfiwDQN
from atari_wrapper import wrap_deepmind, wrap_ram


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256, 256, 256])
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--fast-buffer-size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=50)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=None)
    parser.add_argument('--tper_weight', type=float, default=0.6)
    parser.add_argument('--bk_step', action='store_true')
    parser.add_argument('--reweigh_type', 
                        choices=['linear', 'adaptive_linear', 'done_cnt_linear', 'hard', 'oracle'], 
                        default='hard')
    parser.add_argument("--linear_hp", type=float, nargs='*', default=[0.5, 1.5, 3., -0.3])
    parser.add_argument('--adaptive_scheme', type=float, nargs="*", default=[0.4, 0.8, 1.2, 1.6, 5e6, 1e7])
    parser.add_argument('--lfiw', action='store_true')
    parser.add_argument('--lfiw_temp', type=float, default=0.03)
    parser.add_argument('--lfiw_loss_coeff', type=float, default=0.03)
    return parser.parse_args()


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)
def make_ram_env(args):
    return wrap_ram(args.task, frame_stack=args.frames_stack)

def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)
def make_ram_env_watch(args):
    return wrap_ram(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)

class StepPreprocess():
    
    def __init__(self, buffer_num, bk_step) -> None:
        self.n = buffer_num
        self.cur_traj_step = np.array([0] * buffer_num)
        self.cur_done_cnt = np.array([0] * buffer_num)
        self.bk_step = bk_step
    def get_step(self, **kwargs):
        if 'done' in kwargs:
            dones = kwargs["done"]
            self.cur_traj_step = np.where(dones, 0, self.cur_traj_step + 1)
            self.cur_done_cnt = np.where(dones, self.cur_done_cnt + 1, self.cur_done_cnt)
            return Batch(step=np.array([0] * self.n) if \
                         self.bk_step else self.cur_traj_step,
                         done_cnt=self.cur_done_cnt)
        return Batch()

def test_dqn(args=get_args()):
    if 'ram' in args.task and 'NoFrame' not in args.task:
        use_ram = True
    else:
        use_ram = False
    
    if use_ram:
        env = make_ram_env(args)
        make_env_fn = make_ram_env
        make_watch_fn= make_ram_env_watch
        save_only_last_obs = False
    else:
        env = make_atari_env(args)
        make_env_fn = make_atari_env
        make_watch_fn = make_atari_env_watch
        save_only_last_obs = True

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_env_fn(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_watch_fn(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    if use_ram:
        net = RamDQN(args.state_shape, 
                     args.action_shape, 
                     hidden_sizes=args.hidden_sizes,
                     device=args.device).to(args.device)
    elif args.lfiw:
        net = LfiwDQN(*args.state_shape,
                args.action_shape, args.device).to(args.device)
    else:
        net = DQN(*args.state_shape,
                args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # possible TODO: lfiw_optim over non-cnn parameters
    # prepare hyperparameters
    adaptive_scheme = args.adaptive_scheme
    adaptive_scheme[4] *= args.update_per_step
    adaptive_scheme[5] *= args.update_per_step
    reweigh_hyper = {
        "hard_weight": args.tper_weight,
        "linear": args.linear_hp,
        "adaptive_linear": args.adaptive_scheme,
    }
    # define policy
    if args.lfiw:
        policy = LfiwTPDQNPolicy(net, optim, args.gamma, args.n_step,
                        target_update_freq=args.target_update_freq,
                        bk_step=args.bk_step,
                        reweigh_type=args.reweigh_type,
                        reweigh_hyper=reweigh_hyper,
                        opd_temperature=args.lfiw_temp,
                        opd_loss_coeff=args.lfiw_loss_coeff)
    else:
        policy = TPDQNPolicy(net, optim, args.gamma, args.n_step,
                        target_update_freq=args.target_update_freq,
                        bk_step=args.bk_step,
                        reweigh_type=args.reweigh_type,
                        reweigh_hyper=reweigh_hyper)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if args.lfiw:
        buffer = TPDoubleVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs), bk_step=args.bk_step,
            ignore_obs_next=True,
            save_only_last_obs=save_only_last_obs, 
            stack_num=args.frames_stack,
            fast_buffer_size=args.fast_buffer_size
            )
    else:
        buffer = TPVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs), bk_step=args.bk_step,
            ignore_obs_next=True,
            save_only_last_obs=save_only_last_obs, stack_num=args.frames_stack)
    # collector
    train_collector = Collector(
        policy, 
        train_envs, 
        buffer, 
        preprocess_fn=StepPreprocess(len(train_envs), args.bk_step).get_step,
        exploration_noise=True
    )
    # print(len(test_envs))
    test_collector = Collector(
        policy, 
        test_envs, 
        exploration_noise=True,
    )
    # log
    cur_time = time.strftime('%y-%m-%d-%H-%M-%S', time.localtime())
    log_path = os.path.join(args.logdir, args.task, 'tpdqn', "%s-seed%d"%(args.exp, args.seed), cur_time)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        # if env.env.spec.reward_threshold:
        #     return mean_rewards >= env.spec.reward_threshold
        # elif 'Pong' in args.task:
        #     return mean_rewards >= 20
        # else:
        #     return False
        return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        logger.write('train/eps', env_step, eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = TPVectorReplayBuffer(
                args.buffer_size, buffer_num=len(test_envs),
                ignore_obs_next=True, save_only_last_obs=True,
                stack_num=args.frames_stack)
            collector = Collector(policy, test_envs, buffer)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num,
                                            render=args.render)
        pprint.pprint(result)

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, logger=logger,
        update_per_step=args.update_per_step, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
