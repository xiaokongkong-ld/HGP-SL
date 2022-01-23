import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
from data_process_ABIDE import matrix_filter_topK, matrix_filter_value, matrix_filter_percentage


def plot_mat(matrix, title):
    x, _ = matrix.shape
    plt.matshow(matrix, cmap=plt.cm.Reds)
    plt.title(title)
    plt.show()

def mat_tri(mat):
    x, _ = mat.shape
    tri = np.zeros([x, x])
    for i in range(x):
        for j in range(x):
            if mat[i][j] == 1:
                tri[j][i] = 1
                tri[i][j] = 1
    return tri

def visualize(out, color, center):
    out = np.vstack((out, center))
    leng = len(color)

    z = TSNE(n_components=2).fit_transform(out)
    # z = TSNE(n_components=2).fit_transform(out.detach().numpy())
    data = z[:leng, :]
    centers = z[leng:, :]
    # print(data.shape)
    # print(centers.shape)
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    markers = []
    for lab in color:
        if lab == 0:
            markers.append('r')
        elif lab == 1:
            markers.append('y')
        else:
            markers.append('b')

    plt.scatter(data[:, 0], data[:, 1], s=70, c=markers, cmap="Set2"
                , marker='+'
                )
    plt.scatter(centers[:, 0], centers[:, 1], s=120, c='pink', cmap="Set2"
                , marker='8'
                , alpha=0.6
                )

    plt.show()


def plot_graph(edge_index):
    G = nx.Graph()

    G.add_edges_from(edge_index)
    de = dict(G.degree)

    nx.draw_networkx(G
                     # , node_size=[v * 10 for v in de.values()]
                     , node_size=10
                     , node_color="pink"
                     , node_shape="o"
                     , alpha=0.3
                     , with_labels=False
                     )
    # 绘制网络G
    plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
    plt.show()


def visualize_simple(out):
    z = TSNE(n_components=2).fit_transform(out)
    # z = TSNE(n_components=2).fit_transform(out.detach().numpy())
    data = z

    print(data.shape)

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(data[:, 0], data[:, 1], s=70, c='pink', cmap="Set2"
                , marker='8'
                )

    plt.show()


def visualize_ori(out, lab):
    z = TSNE(n_components=2).fit_transform(out)
    # z = TSNE(n_components=2).fit_transform(out.detach().numpy())
    data = z

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(data[:, 0], data[:, 1], s=70, c=lab, cmap="Set2"
                , marker='8'
                )

    plt.show()


def visualize_with_edge(out, edge):
    x, y = out.shape
    print(out.shape)
    z = TSNE(n_components=2).fit_transform(out)
    # z = TSNE(n_components=2).fit_transform(out.detach().numpy())
    center_len = x - y
    data = z
    list_x = []
    list_y = []
    for i, o in enumerate(edge):
        ax = data[o[0]][0]
        ay = data[o[0]][1]
        bx = data[o[1]][0]
        by = data[o[1]][1]

        x_t = [ax, bx]
        y_t = [ay, by]
        list_x.append(x_t)
        list_y.append(y_t)

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    for i in range(len(list_x)):
        # plt.plot(list_x[i], list_y[i], color='pink', marker='8')
        if i < 65:
            plt.plot(list_x[i], list_y[i], color='red', marker='8')
        else:
            plt.plot(list_x[i], list_y[i], color='green', marker='8', alpha=0.3)
    # plt.show()
    #
    plt.scatter(data[:, 0], data[:, 1], s=120, c='pink', cmap="Set2"
                , marker='+'
                )
    plt.show()


if __name__ == "__main__":
    print('hello')

    # root_matrix = np.load('./abide_built/autism_400_hierarchy/autism_hierarchy_15_root_matrix.npy')
    # data_matrix = np.load('./abide_built/autism_400_hierarchy/autism_hierarchy_15_new_matrix.npy')
    # root_matrix = np.load('./abide_built/test/control_adj_68.npy')
    # data_matrix = np.load('./abide_built/test/autism_hierarchy_6_new_matrix.npy')
    # root_matrix = np.load('./abide_built/test/autism_hierarchy_88_mix_root_matrix.npy')

    # root_matrix = np.load('./ABIDE/matrix/autism_400/autism_400_6.npy')
    # root_matrix = np.load('./abide_built/test/autism_hierarchy_6_root_matrix.npy')
    # root_matrix = np.load('./abide_built/autism_400_hierarchy/autism_hierarchy_66_root_matrix.npy')
    # root_matrix = matrix_filter_topK(root_matrix, 2)
    # root_matrix = matrix_filter_value(root_matrix, 0.1)
    # root_matrix = matrix_filter_percentage(root_matrix, 0.01)
    # root_matrix1 = matrix_filter_topK(root_matrix, 16)
    # root_matrix2 = matrix_filter_value(root_matrix, 0.1)
    # root_matrix3 = matrix_filter_percentage(root_matrix, 0.15)
    # root_matrix = root_matrix1 + root_matrix2 + root_matrix3
    # root_matrix = matrix_filter_value(root_matrix, 3)

    # plot_mat(root_matrix, 'root matrix')
    # plot_mat(data_matrix, 'data matrix')
    # plot_mat(data_ts, 'data ts')

    # o = coo_matrix(root_matrix)
    # e = np.vstack((o.row, o.col)).T
    # print(e)
    #
    # visualize_with_edge(data_matrix, e)

    # plot_graph(e)

    # mat_ori = np.load('./abide_built/control_400_ts/control_2_new_ts.npy')
    # mat_tsne = np.load('./abide_built/control_400_tsne/control_2_tsne.npy')
    #
    # print(mat_ori.shape)
    # print(mat_tsne.shape)
    # mat_ori = pd.DataFrame(mat_ori.T)
    # mat_tsne = pd.DataFrame(mat_tsne.T)
    #
    # print(mat_ori)
    # print(mat_tsne)
    #
    # corr_mat_ori = mat_ori.corr()
    # corr_mat_tsne = mat_tsne.corr()
    #
    #
    # # f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corr_mat_ori, square=True)
    # # sns.heatmap(corr_mat_tsne, square=True)
    # plt.show()
    # ed = np.array(
    #     [[1, 2], [2, 3], [2, 4], [3, 5], [3, 6], [4, 7], [4, 8], [5, 9], [5, 10], [2, 0], [6, 11], [6, 12], [7, 13],
    #      [7, 14], [8, 15], [8, 16],[16, 17],[16, 18],[16, 19],[16, 20], [9, 10], [17,18], [17, 19],[19,20], [18,19]])
    # print(type(ed))
    # G = nx.Graph()
    #
    # G.add_edges_from(ed)
    # hyp = hyperbolicity_sample(G)
    # plot_graph(ed)
    # print('Hyp: ', hyp)
    # mat = np.load('./adj.npy')
    # mat0 = mat[:400, :]
    # mat9 = mat[3200:3600, :]
    # tri9 = mat_tri(mat9)
    #
    # print(mat[:800, :].shape)
    # plot_mat(mat0, 'adj')
    # plot_mat(mat9, '9')
    # plot_mat(tri9,'tri9')
    #
    # o = coo_matrix(mat0)
    # e = np.vstack((o.row, o.col)).T
    #
    # plot_graph(e)
    #
    # G = nx.Graph()
    #
    # G.add_edges_from(e)
    # hyp = hyperbolicity_sample(G)
    # print('Hyp: ', hyp)


