import dgl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path_object = 'data_dgl/object.csv'
path_object_object = 'data_dgl/object-interact-object.csv'
path_room = 'data_dgl/room.csv'
path_room_object = 'data_dgl/room-interact-object.csv'
path_attribute = 'data_dgl/attribute.csv'
path_attribute_object = 'data_dgl/attribute-behave-object.csv'

# https://networkx.github.io/documentation/stable/tutorial.html#drawing-graphs
def load_heterograph():
    nodes_feature_object = read_node_data(path_object)
    nodes_feature_room = read_node_data(path_room)
    nodes_feature_attribute = read_node_data(path_attribute)
    #
    src_objectobject, dst_objectobject = read_edge(path_object_object)
    edges_objectobject = list(map(lambda tup: tup, zip(
        src_objectobject, dst_objectobject)))
    #
    src_roomobject, dst_roomobject = read_edge(path_room_object)
    start_id_room = nodes_feature_object.shape[0]
    src_roomobject += start_id_room
    edges_roomobject = list(map(lambda tup: tup, zip(
        src_roomobject, dst_roomobject)))
    #
    src_attributeobject, dst_attributeobject = read_edge(path_attribute_object)
    start_id_attribute = start_id_room + nodes_feature_room.shape[0]
    src_attributeobject += start_id_attribute
    edges_attributeobject = list(map(lambda tup: tup, zip(
        src_attributeobject, dst_attributeobject)))

    G = nx.MultiDiGraph()

    G.add_edges_from(edges_objectobject)
    G.add_edges_from(edges_roomobject)
    G.add_edges_from(edges_attributeobject)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    # import pdb;pdb.set_trace()
    # G.nodes
    # G.add_nodes_from([3, 2], time=["2pm", "3"])
    # {'time': ['2pm', '3']}
    # G.nodes[3]
    # G.edges
    return G


def load_homogeneous(path_node, path_edge):
    # edges
    node_feature_object = read_node_data(path_node)
    src_node1_node2, dst_node1_node2 = read_edge(path_edge)
    g = dgl.graph((src_node1_node2, dst_node1_node2))
    g.ndata['feature'] = node_feature_object
    nx_g = dgl.to_networkx(g, node_attrs=['feature'])
    return nx_g


def read_node_data(path):
    def get_feature(csv_nodes_data):
        feature = [csv_nodes_data['feature'].to_list()]
        for i in range(1, 300):
            feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
        feature = torch.tensor(feature).float().transpose(0, 1)
        return feature

    nodes_data = pd.read_csv(path)
    node_feature = get_feature(nodes_data)
    return node_feature


def read_edge(path):
    edges_data = pd.read_csv(path)
    src_node1_node2 = edges_data['Src'].to_numpy()
    dst_node1_node2 = edges_data['Dst'].to_numpy()
    return src_node1_node2, dst_node1_node2


def draw_graph(G):
    # import pdb; pdb.set_trace()
    # plt.subplot(121)
    pos = nx.spring_layout(G)

    nx.draw(G, with_labels=True, font_weight='bold', pos=pos)
    # plt.subplot(122)

    # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold', pos=pos)
    plt.show()


if __name__ == '__main__':
    # ('object', 'interacts', 'object')
    # g1 = load_homogeneous(path_object, path_object_object)
    # load_heterograph
    g_multi = load_heterograph()
    draw_graph(g_multi)
