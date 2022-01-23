import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import HGP, GCN_ori, HGCN_pyg, GCN_adj
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# from data_process_unknow import get_data, create_dataset
from data_process_ABIDE import get_data, create_dataset, train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES/OHSU/Peking_1/KKI')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0/cpu')
parser.add_argument('--epochs', type=int, default=45, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--c', type=int, default=1, help='radiu')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
################################# TUdataset ###########################################
# dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
#
# args.num_classes = dataset.num_classes
# args.num_features = dataset.num_features
#
# print(f'Dataset {args.dataset}:')
# print('====================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {args.num_features}')
# print(f'Number of classes: {args.num_classes}')

############################### brain dataset unknow ##################################
# train_data = get_data("Training")
# train_data_features = get_data("Training")
#
# test_data = get_data("Testing")
# test_data_features = get_data("Testing")
#
# training_dataset = create_dataset(train_data, features=train_data_features)
# testing_dataset = create_dataset(test_data, features=test_data_features)
#
# # After getting the Datasets we load the data into the respective Dataloader
#
# train_loader = DataLoader(training_dataset, batch_size=5, shuffle=True)
# test_loader = DataLoader(testing_dataset, batch_size=5, shuffle=True)
#
# args.num_classes = 2
# args.num_features = 150

############################### brain dataset unknow ###################################
groups = ['autism', 'control']
subs_data = get_data(groups)
dataset = create_dataset(subs_data, 4)

# training_set, test_set = train_test_split(dataset, 0.8)
#
# train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

args.num_classes = 2
args.num_features = 400
########################################################################################

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

####################################### models ########################################

# model = HGP(args).to(args.device)
model = GCN_ori(args).to(args.device)
# model = HGCN_pyg(args).to(args.device)
# model = GCN_adj(args).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)

            # out = model(data)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        # acc_test, loss_test = compute_test(test_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.3f}'.format(loss_train),
              'acc_train: {:.3f}'.format(acc_train), 'loss_val: {:.3f}'.format(loss_val),
              'acc_val: {:.3f}'.format(acc_val))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.3f}, accuracy = {:.3f}'.format(test_loss, test_acc))

