import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
args = parser.parse_args()

with open("log/MiniGrid-Empty-8x8-v0/dqn/Q_table_onpolicypickle%d"%args.n, 'rb') as f:
    Q_table = pickle.load(f)

V_table = {}
for key, value in zip(Q_table.keys(), Q_table.values()):
    V_table[key] = np.max(value)
V_mean = np.average(list(V_table.values()))

V_loss_table = []
V_loss_linear = {}
for i in range(11):
    V_loss_linear[i] = []
for i in range(1, 7):
    this_loss = []
    for j in range(1, 7):
        real_V = 0.99 ** ((6-i) + (6-j))
        try:
            loss = abs(V_table[(i,j)] - real_V)
        except KeyError:
            # loss = abs(V_mean - real_V)
            loss = 0
        this_loss.append(loss)
        V_loss_linear[12-i-j].append(loss)
    V_loss_table.append(this_loss)
V_loss_table = np.array(V_loss_table)
# V_loss_table = np.array(V_loss_table)[:4, :4]
for i in range(11):
    V_loss_linear[i] = np.average(V_loss_linear[i])
# print(V_loss_linear)
sns.heatmap(V_loss_table, cmap="YlGnBu", vmin=0.2, vmax=0.7)
# sns.heatmap(V_loss_table, cmap="YlGnBu")
plt.savefig("log/MiniGrid-Empty-8x8-v0/all-grid-headmap-%d"%args.n)
