import random
import numpy as np
import torch
# torch.set_printoptions(profile="full")
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import os
import heapq
import networkx as nx
import matplotlib.pyplot as plt

DATA_PATH = './data_brain/abide'
groups = ['autism', 'control']
atlas = ['_aal', '_schaefer100', '_schaefer400', '_schaefer1000']


def get_data(group_list):
    """ read .npy data from disk by group list and put into subject list"""
    sub_list = []
    group_number = len(group_list)
    for i in range(group_number):
        subs_data = load_data(group_list[i])
        count = 0
        for sub in subs_data:
            count += 1
            sub_list.append([sub, i])
    return sub_list


def create_dataset(data, top_K):
    dataset_list = []
    for i in range(len(data)):
        correlation_matrix_f = matrix_filter_topK(data[i][0], top_K)
        edge_index_coo = coo_matrix(correlation_matrix_f)
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)

        feature_matrix = data[i][0]
        graph_data = Data(x=torch.tensor(feature_matrix, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(data[i][1]))
        dataset_list.append(graph_data)
        # if i > 1:
        #     break
    return dataset_list


def load_data(group):
    """ read .npy data from disk """
    data_path = os.path.join(DATA_PATH)
    load_name = '/' + group + atlas[2] + '.npy'
    data_path = data_path + load_name
    data = np.load(data_path)
    print(f'load data from: {data_path}')
    return data


def show_subs_labels(data):
    """ show data and its label"""
    sub_list_len = len(data)
    print(f'we have {sub_list_len} subs')
    print('labels are:')
    for i in range(sub_list_len):
        label = data[i][1]
        group = groups[label]
        print(f'number: {i + 1}, group: {group}')


def matrix_filter_percentage(mat, perc):
    x, y = mat.shape
    mac = mat.copy()
    np.fill_diagonal(mac, -10)
    mac = mac.reshape(-1)
    k = int(len(mac) * perc)
    idx = mac.argsort()
    idx = idx[::-1]
    top_k_idx = idx[:k]
    down_k_id = idx[k:]
    mac[top_k_idx] = 1
    mac[down_k_id] = 0
    return mac.reshape(x, -1)


def matrix_filter_value(matrix, filter):
    f_mat = matrix.copy()
    np.fill_diagonal(f_mat, 0)
    f_mat2 = f_mat.copy()
    f_mat2[f_mat >= filter] = 1
    f_mat2[f_mat < filter] = 0
    return f_mat2


def matrix_filter_topK(matrix, K):
    """ filt matrix with top K max"""
    m_l = len(matrix[0])
    index = [x for x in range(m_l)]
    f_mat = np.zeros(m_l)
    mat = matrix.copy()
    np.fill_diagonal(mat, -100)

    for x in mat:
        y = heapq.nlargest(K, range(len(x)), x.take)
        y_else = np.setdiff1d(index, y)
        x[y] = 1
        x[y_else] = 0
        f_mat = np.vstack((f_mat, x))
    f_mat = np.delete(f_mat, np.s_[0], axis=0)
    # print('=====================================================================================')
    # print(f_mat)
    return f_mat


def plot_mat(matrix, title):
    x, _ = matrix.shape
    plt.matshow(matrix, cmap=plt.cm.Reds)
    plt.title(title)
    plt.show()


def train_test_split(data, train_percent):
    sub_train = []
    sub_test = []
    num_sub = len(data)
    node = [x for x in range(num_sub)]
    train_id = random.sample(node, int(train_percent * num_sub))
    for x in train_id:
        node.remove(x)
    test_id = node

    for i in train_id:
        sub_train.append(data[i])
    for j in test_id:
        sub_test.append(data[j])

    return sub_train, sub_test


def show_dataset(dataset, datatype):
    print(f'.....................................{datatype} dataset ............................................')
    for i in range(len(dataset)):
        print(dataset[i])


def symatrix_2_halfmatrix(mat):
    lenth, _ = mat.shape
    matrix = mat.copy()
    for i in range(lenth):
        for j in range(i, lenth):
            matrix[i][j] = 0
            # print(matrix[i][j])
    return matrix


def asymatrix_2_halfmatrix(mat):
    lenth, _ = mat.shape
    matrix = mat.copy()
    for i in range(lenth):
        for j in range(i, lenth):
            if matrix[i][j] == 1:
                matrix[j][i] = 1
                matrix[i][j] = 0
    return matrix


def plot_graph(edge_index):
    G = nx.Graph()

    G.add_edges_from(edge_index)
    de = dict(G.degree)

    nx.draw_networkx(G
                     # , pos
                     , node_size=[v * 10 for v in de.values()]
                     , node_color="blue"
                     , node_shape="o"
                     , alpha=0.3
                     , with_labels=False
                     )
    # 绘制网络G
    plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
    plt.show()


if __name__ == "__main__":
    print('hello')
    # subs_data = get_data(groups)
    # subs = create_dataset(subs_data, 2)
    #
    # show_dataset(subs, 'all')
    # # a = np.array([[0,0,1,1],[0,1,0,1],[1,1,1,0],[0,0,0,1]])
    # # o = coo_matrix(a)
    # # e = torch.tensor(np.vstack((o.row, o.col)), dtype=torch.long)
    # # e = e.numpy().T
    # # print(e)
    # e = np.load('./edge/edge_index_6.npy')
    # print(len(e))
    # plot_graph(e)
    data = load_data(groups[0])

    for i, d in enumerate(data):
        if i > 1:
            break
        d = torch.tensor(d, dtype=torch.float32)
        plot_mat(d, 'before mat')
        ds = torch.sigmoid(d)
        plot_mat(ds, 'after mat')

    s = torch.tensor([[0.2, 0.2, 0.3], [0.4, 0.1, 0.2], [0.5, 0.4, 0.7]], dtype=torch.float32)
    print(s)
    plot_mat(s, 'before')
    ss = torch.tanh(s - 0.5)
    plot_mat(ss, 'after')
    print(ss)
