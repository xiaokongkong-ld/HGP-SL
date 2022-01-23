import random
import numpy as np
import torch

torch.set_printoptions(profile="full")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix


def get_data(dataset="Training"):
    healthy = []
    patient = []

    if dataset == "Training":
        for i in range(1, 11):
            healthy.append(
                [np.genfromtxt('./data_brain/unknow/' + dataset + '/Health/sub' + str(i) + '/common_fiber_matrix.txt'),
                 np.genfromtxt(
                     './data_brain/unknow/' + dataset + '/Health/sub' + str(i) + '/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append(
                [np.genfromtxt('./data_brain/unknow/' + dataset + '/Patient/sub' + str(i) + '/common_fiber_matrix.txt'),
                 np.genfromtxt(
                     './data_brain/unknow/' + dataset + '/Patient/sub' + str(i) + '/pcc_fmri_feature_matrix_0.txt'), 0])

    elif dataset == "Testing":
        for i in range(1, 6):
            healthy.append(
                [np.genfromtxt('./data_brain/unknow/' + dataset + '/Health/sub' + str(i) + '/common_fiber_matrix.txt'),
                 np.genfromtxt(
                     './data_brain/unknow/' + dataset + '/Health/sub' + str(i) + '/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append(
                [np.genfromtxt('./data_brain/unknow/' + dataset + '/Patient/sub' + str(i) + '/common_fiber_matrix.txt'),
                 np.genfromtxt(
                     './data_brain/unknow/' + dataset + '/Patient/sub' + str(i) + '/pcc_fmri_feature_matrix_0.txt'), 0])

    data = []
    for i in range(len(healthy)):
        data.append(healthy[i])
        data.append(patient[i])

    del healthy, patient
    return data


# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset
def create_dataset(data, features):
    dataset_list = []
    for i in range(len(data)):
        edge_index_coo = coo_matrix(data[i][0])

        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)

        adj_mat = torch.tensor(data[i][0], dtype=torch.float32)

        feature_matrix = features[i][1]
        graph_data = Data(x=torch.tensor(feature_matrix, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(data[i][2]), adj=adj_mat)
        dataset_list.append(graph_data)
    return dataset_list


train_data = get_data("Training")
train_data_features = get_data("Training")

test_data = get_data("Testing")
test_data_features = get_data("Testing")

training_dataset = create_dataset(train_data, features=train_data_features)
testing_dataset = create_dataset(test_data, features=test_data_features)

# Afetr getting the Datasets we load the data into the respective Dataloader

train_loader = DataLoader(training_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=5, shuffle=True)
