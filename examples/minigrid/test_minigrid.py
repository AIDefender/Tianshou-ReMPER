import gym
import os
import gym_minigrid
from gym_minigrid.wrappers import *
from atari_network import DQN
import argparse
import torch
from tianshou.policy import DQNPolicy
import pickle
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=None)
    parser.add_argument('-n', type=int)
    return parser.parse_args()

def make_minigrid_env():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    return env

def eval_minigrid(args):
    device = 'cuda'
    env = make_minigrid_env()
    state_shape = env.observation_space.shape
    action_shape = env.env.action_space.n
    net = DQN(state_shape[2], state_shape[0], state_shape[1],
              action_shape, device).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        path = os.path.join(args.resume_path, "policy-%d.pth"%args.n)
        policy.load_state_dict(torch.load(path, map_location=device))
        print("Loaded agent from: ", path)

    env.reset()
    action = None
    Q_table = {}
    i = 0
    while True:
        if action is None:
            action = 4
        action = np.random.randint(3)
        state, reward, done, _ = env.step(action)
        pos = tuple(env.agent_pos)
        value = net(state.reshape(1, state_shape[0], state_shape[1], state_shape[2]))[0].detach().cpu().numpy()
        # action = np.argmax(value)
        Q_table[pos] = value
        print(i, pos, value)
        # if done or i > 10000:
        if i > 3000:
            break
        i += 1

    with open("log/MiniGrid-Empty-8x8-v0/dqn/Q_table_onpolicy%d.txt"%args.n, 'w') as f:
        for value, key in zip(Q_table.values(), Q_table.keys()):
            print(key, ":", value, file=f)
    with open("log/MiniGrid-Empty-8x8-v0/dqn/Q_table_onpolicypickle%d"%args.n, 'wb') as f:
        pickle.dump(Q_table, f)

args = get_args()
eval_minigrid(args)