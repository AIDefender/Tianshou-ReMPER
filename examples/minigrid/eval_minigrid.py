import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
import os
sns.set_context('paper', font_scale=1.5)
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
parser.add_argument('--resume-path', type=str, default=None)
parser.add_argument('--title', type=str, default='default')
args = parser.parse_args()

def get_mask():
    mask = np.zeros((7, 7))
    mask[3, :] = 1
    mask[:, 3] = 1
    mask[3, 1] = 0
    mask[3, 5] = 0
    mask[5, 3] = 0
    mask[0, 6] = 1
    mask[0, 0] = 1

    return mask
all_V_loss = []
for seed in os.listdir(args.resume_path):
    if seed.startswith("heatmap"):
        continue
    with open(os.path.join(args.resume_path, str(seed), "Q_tablepickle%d"%args.n), 'rb') as f:
        Q_table = pickle.load(f)
    print("Loaded Q table ", os.path.join(args.resume_path, "Q_tablepickle%d"%args.n))

    V_table = {}
    for key, value in zip(Q_table.keys(), Q_table.values()):
        V_table[key] = np.max(value)
    V_mean = np.average(list(V_table.values()))

    V_loss_table = []
    V_loss_linear = {}
    for i in range(14):
        V_loss_linear[i] = []
    for i in range(1, 8):
        this_loss = []
        for j in range(1, 8):
            real_V = 0.99 ** ((6-i) + (6-j))
            try:
                loss = abs(V_table[(i,j)] - real_V)
            except KeyError:
                # loss = abs(V_mean - real_V)
                loss = 1
            this_loss.append(loss)
            V_loss_linear[14-i-j].append(loss)
        V_loss_table.append(this_loss)
    V_loss_table = np.array(V_loss_table)
    all_V_loss.append(V_loss_table)
mean_V_loss = np.average(all_V_loss, axis=0)


# ===========plot=============
fig, ax = plt.subplots()

# frame = sns.heatmap(mean_V_loss, cmap="YlGnBu", vmin=0.1, vmax=0.5)
frame = sns.heatmap(mean_V_loss, cmap = 'rocket_r', vmin=0.15, vmax=0.7, mask=get_mask(), ax=ax)
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
frame.set_facecolor("black")
triangle = plt.imread('examples/minigrid/fig/triangle.png')
square = plt.imread('examples/minigrid/fig/square.png')
newax = fig.add_axes([0.65, 0.78, 0.1, 0.1])
newax.imshow(square)
newax.set_xticks([])
newax.set_yticks([])

newax2 = fig.add_axes([0.12, 0.78, 0.1, 0.1])
newax2.imshow(triangle)
newax2.set_xticks([])
newax2.set_yticks([])

# =========save fig============
if not os.path.isdir(os.path.join(args.resume_path, "heatmap")):
    os.mkdir(os.path.join(args.resume_path, "heatmap"))
fig.suptitle(args.title)
plt.savefig(os.path.join(args.resume_path, "heatmap", "%d.png"%args.n))
