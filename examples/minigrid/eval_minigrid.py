import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
import os
sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
parser.add_argument('--resume-path', type=str, default=None)
args = parser.parse_args()

with open(os.path.join(args.resume_path, "Q_tablepickle%d"%args.n), 'rb') as f:
    Q_table = pickle.load(f)

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
# V_loss_table = np.array(V_loss_table)[:4, :4]
for i in range(11):
    V_loss_linear[i] = np.average(V_loss_linear[i])
# print(V_loss_linear)
sns.heatmap(V_loss_table, cmap="YlGnBu", vmin=0.2, vmax=0.7)
# sns.heatmap(V_loss_table, cmap="YlGnBu")
plt.savefig(os.path.join(args.resume_path, "heatmap", "%d.png"%args.n))
