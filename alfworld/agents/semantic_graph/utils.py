import os
import glob
import json
import numpy as np
from torch_geometric.utils import to_networkx
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
'''
# https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
# https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=Y9MOs8iSwKFD
G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)
# https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=9r_VmGMukf5R
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
'''


def visualize_node(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
        # h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
        #                  node_color=color, cmap="Set2")
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    return plt


def visualize_node_feature(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    return plt


def visualize_points(pos, edge_index=None, index=None):
    pos = TSNE(n_components=2).fit_transform(pos.detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
        mask = torch.zeros(pos.size(0), dtype=torch.bool)
        mask[index] = True
        plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
        plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()
    return plt


def save_graph_data(graph_data, path):
    num_obj_cls = len(graph_data.obj_cls_to_features.keys()) + 1
    colors = cm.rainbow(np.linspace(0, 1, num_obj_cls))
    node_to_color = [colors[node_obj_cls_ind] for node_obj_cls_ind in graph_data.list_node_obj_cls]
    # G = to_networkx(graph_data, to_undirected=False)
    plt = visualize_node(graph_data.x, color=node_to_color)
    save_plt_img(plt, path, "node_")
    plt = visualize_node_feature(graph_data.x, color=node_to_color)
    save_plt_img(plt, path, "node_feature_")
    plt = visualize_points(graph_data.x, edge_index=graph_data.edge_obj_to_obj)
    save_plt_img(plt, path, "node_points_")


def save_plt_img(plt, path, name):
    idx = len(glob.glob(path + '/{}*.png'.format(name)))
    name = os.path.join(path, '%s%09d.png' % (name, idx))
    plt.savefig(name)


def load_graph_data(path):
    pass


def save_final_dynamics(path, final_dynamics, name="final_dynamics"):
    final_dynamics_path = os.path.join(path, name + ".json")
    for key in final_dynamics.keys():
        final_dynamics[key]["final_dynamics"] = final_dynamics[key]["final_dynamics"].tolist()
        if "observation_feats" in final_dynamics[key]:
            final_dynamics[key]["observation_feats"] = final_dynamics[key]["observation_feats"].tolist()[0]
    with open(final_dynamics_path, 'w') as f:
        json.dump(final_dynamics, f)
    data_to_t_sne(path, final_dynamics, name=name, s_feature="final_dynamics")
    if "observation_feats" in final_dynamics[key]:
        data_to_t_sne(path, final_dynamics, name=name, s_feature="observation_feats")


def data_to_t_sne(path, data, name="output", s_feature="final_dynamics"):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import pandas as pd
    import seaborn as sns
    X = None
    label = []
    for key in data.keys():
        label.append(data[key]["label"])
        feature = data[key][s_feature]
        feature = np.array([feature])
        if X is None:
            X = feature
        else:
            X = np.concatenate((X, feature), axis=0)
    X = PCA(n_components=50).fit_transform(X)
    for perplexity in [10, 20, 30, 40, 50]:
        tsne_results = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=2000).fit_transform(X)
        df = pd.DataFrame()
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        df['label'] = label
        sns_plot = sns.lmplot(
            data=df,
            x='tsne-2d-one',
            y='tsne-2d-two',
            hue='label',
            fit_reg=False
        )
        fig_name = os.path.join(path, name + "_" + s_feature + "_%d.png" % perplexity)
        sns_plot.savefig(fig_name)


if __name__ == '__main__':
    import ast
    path = "D:\\alfred\\alfworld\\videos"
    file = "D:\\alfred\\alfworld\\videos\\valid_seen_final_dynamics.json"
    name = "valid_seen_final_dynamics"

    f = open(file, "r")
    data = f.read()
    data = ast.literal_eval(data)
    for key in data.keys():
        final_dynamics = data[key]["final_dynamics"]
        data[key]["final_dynamics"] = np.array(final_dynamics)
        if "observation_feats" in data[key]:
            observation_feats = data[key]["observation_feats"]
            data[key]["observation_feats"] = np.array(observation_feats)[0]
    data_to_t_sne(path, data, name=name)
    if "observation_feats" in data[key]:
        data_to_t_sne(path, data, name=name, s_feature="observation_feats")

    # file = "D:\\alfred\\alfworld\\videos\\valid_unseen_final_dynamics.json"
    # f = open(file, "r")
    # data = f.read()
    # data = ast.literal_eval(data)
    # for key in data.keys():
    #     final_dynamics = data[key]["final_dynamics"]
    #     data[key]["final_dynamics"] = np.array(final_dynamics)
    #     if "observation_feats" in final_dynamics[key]:
    #         observation_feats = data[key]["observation_feats"]
    #         final_dynamics[key]["observation_feats"] = np.array(observation_feats)

    # data_to_t_sne("D:\\alfred\\alfworld\\videos", data, name="valid_unseen_final_dynamics")
    # if "observation_feats" in final_dynamics[key]:
    #     data_to_t_sne(path, final_dynamics, name="valid_unseen_final_dynamics", feature="observation_feats")