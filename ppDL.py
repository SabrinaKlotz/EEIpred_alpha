#%%
import os
import csv
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Post-processing parameters")

parser.add_argument(
    "-c", "--cuda", type=str, help="number of GPU", required=False, default="0"
)
parser.add_argument(
    "-mth", "--method", type=str, help="RRI method (dMaSIF,PInet,glinter,ProteinMAE)", required=True
)
parser.add_argument(
    "-d", "--dataset", type=str, help="dataset (CONTACT,EPPIC,PISA)", required=True
)
parser.add_argument(
    "-md", "--mode", type=str, help="train or test", required=False, default="test"
)
parser.add_argument(
    "-f", "--folds", type=str, help="folds to use (1,2,3,4,5)", required=True
)
parser.add_argument(
    "-e", "--epochs", type=int, help="number of epochs", required=False, default=20
)
parser.add_argument(
    "-b", "--batch", type=int, help="batch size", required=False, default=128
)
parser.add_argument(
    "-lr", "--learning_rate", type=float, help="learning rate", required=False, default=0.0001
)
parser.add_argument(
    "-s", "--seed", type=int, help="random seed", required=False, default=42
)
parser.add_argument(
    "-tv", "--train_val", type=float, help="train/val split", required=False, default=0.1
)
parser.add_argument(
    "-em", "--eval_modus", type=str, help="save predictions, choose test_set or train_set\n\
        default: test_set", required=False, default="test_set"
)

optn = parser.parse_args()
device = torch.device("cuda:{}".format(optn.cuda) if torch.cuda.is_available() else "cpu")

# dataset class for 100x100 max size AA level data
class ExonDataset(Dataset):
    def __init__(self, data, filename):
        self.data = data

        self.pairs = {}
        #filename = "../../../data_collection/int_exon_pairs.txt"
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exon_name = row['CHAIN1'] + "_" + row['CHAIN2'][-1:] + "_" + \
                    row['EXON1'] + "_" + row['EXON2']
                if exon_name not in self.pairs:
                    self.pairs[exon_name] = 1

    def __getitem__(self, index):
        self.exons = np.load(self.data[index], allow_pickle=True)
        shape_1 = self.exons.shape[0]
        shape_2 = self.exons.shape[1]
        self.exonData = np.zeros((100, 100))
        if shape_1 > 100 or shape_2 > 100:
            print("Exon too big!")
            return self.exonData, 0
        self.exonData[:shape_1, :shape_2] = self.exons[:, :]
        exons = self.data[index].split('_')

        if exons[-6] + "_" + exons[-5] + "_" + exons[-3] + "_" + exons[-2] \
            + "_" + exons[-1][:-4] in self.pairs or \
            exons[-6] + "_" + exons[-3] + "_" + exons[-5] + "_" +\
            exons[-1][:-4] + "_" + exons[-2] in self.pairs:    
            labels = 1
        else:
            labels = 0
        return self.exonData, labels, self.data[index].split("/")[-1].split(".")[0]#name#, self.data[index][:-8]
    
    def __len__(self):
        return len(self.data)

       
# # model for 2D 100x100 images with a small and big kernel 
class BinaryMatrixClassifier(nn.Module):
    def __init__(self):
        super(BinaryMatrixClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128) 
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, x):
        # add channel dimension
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 25 * 25)  # Adjust view size based on your matrix size
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

#%%
# path to exons
if optn.method == "ProteinMAE":
    root_path = "../ProteinMAE/search"
else:
    root_path = optn.method
save_path = "ppDL_models/{}/{}/".format(optn.method, optn.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if len(optn.dataset.split("_")) > 1:
    dataset = optn.dataset.split("_")[1]
else:
    dataset = optn.dataset
exon_labels = "data_collection/cv_splits/{}/{}_positives.txt".format(dataset, dataset)

if optn.mode == "train":
    save_path = save_path + "fold"
    for i in optn.folds.split(","):
        
        cv_idx = int(i)

        # load all files from one folder with glob
        train_folder = "{}/results/{}/fold{}/{}/".format(root_path, optn.dataset, cv_idx,optn.mode)
        print(train_folder)
        train_data = glob.glob(train_folder + "*.npy")
        train_data = [x for x in train_data if "big.npy" not in x]

        train_data, val_data = train_test_split(train_data, test_size=optn.train_val, random_state=optn.seed)

        # create datasets
        trainDataset = ExonDataset(train_data, filename = exon_labels)
        valDataset = ExonDataset(val_data, filename = exon_labels)

        # create dataloaders
        trainLoader = DataLoader(trainDataset, batch_size=optn.batch, shuffle=True)
        valLoader = DataLoader(valDataset, batch_size=optn.batch, shuffle=False)

        model = BinaryMatrixClassifier()
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=optn.learning_rate, amsgrad=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


        # train model
        trainLoss = []
        testLoss = []

        for epoch in range(optn.epochs):
            model.train()
            train_loss = []
            for inputs, labels, pair in tqdm(trainLoader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs.float())#[:,0]

                loss = criterion(outputs.squeeze(1), labels.float())#.squeeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            trainLoss.append(np.mean(np.asarray(train_loss)))
            model.eval()
            with torch.no_grad():
                val_loss = []
                for inputs, labels, pair in tqdm(valLoader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs.float())#[:,0]

                    #labels = labels.float().view(-1, 1)
                    #labels = F.one_hot(labels.to(torch.int64), num_classes=2)
                    loss = criterion(outputs.squeeze(1), labels.float())
                    #loss = criterion(outputs, labels)#.squeeze(1).float())
                    val_loss.append(loss.item())
                testLoss.append(np.mean(np.asarray(val_loss)))
            scheduler.step()
            if testLoss[-1] == min(testLoss):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, save_path + "{}.pth".format(cv_idx))
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, save_path + "{}_{}.pth".format(cv_idx, epoch))
                
            print("Epoch: %d, Train Loss: %.3f, Val Loss: %.3f" % (epoch, trainLoss[epoch], testLoss[epoch]))

# %%
if optn.mode == "test":
    if optn.eval_modus == "":
        print("Please choose test_set or train_set to get predictions!")
        exit()

    for i in optn.folds.split(","):
        results_per_fold = list()
        cv_idx = int(i)
        model_path = save_path + "fold{}.pth".format(cv_idx)
        if not os.path.exists(model_path):
            print("Model for fold {} not found!\nPath given: {}".format(cv_idx, model_path))
            exit()
        # load all files from one folder with glob
        test_folder = "{}/results/{}/fold{}/{}/".format(root_path, optn.dataset, cv_idx,optn.eval_modus.split("_")[0])
        test_data = glob.glob(test_folder + "*.npy")

        test_data = [x for x in test_data if "big.npy" not in x]
        test_data = sorted(test_data)

        testDataset = ExonDataset(test_data, filename = exon_labels)

        # create dataloaders
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

        model = BinaryMatrixClassifier()
        model = model.to(device)
        # load model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        pos = list()
        neg = list()
        with torch.no_grad():
            model.eval()
            counter = 0
            for inputs, labels, pair in tqdm(testLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs.float())
                # convert outputs to numpy at cpu
                outputs = outputs.cpu().numpy()[0]
                if labels == 1:
                    pos.append(outputs[0])
                else:
                    neg.append(outputs[0])
                results_per_fold.append([pair[0], outputs[0], labels[0].item()])

        pos = np.asarray(pos)
        neg = np.asarray(neg)
        if optn.dataset == "EXAMPLE":
            print("pair\t\t\t\t\t\t\tprediction\tlabel")
            for i in results_per_fold:
                print(i[0], i[1], i[2])
         write results_per_fold to file
         for readable format
        if optn.method == "dmasif":
            if not os.path.exists("results/dMaSIF_DL/"):
                os.makedirs("results/dMaSIF_DL/")
            with open("results/dMaSIF_DL/{}_fold{}_results.csv".format(optn.dataset, cv_idx), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "prediction", "label"])
                writer.writerows(results_per_fold)
        elif optn.method == "glinter":
            if not os.path.exists("results/GLINTER_DL/"):
                os.makedirs("results/GLINTER_DL/")
            with open("results/GLINTER_DL/{}_fold{}_results.csv".format(optn.dataset, cv_idx), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "prediction", "label"])
                writer.writerows(results_per_fold)
        elif optn.method == "ProteinMAE":
            if not os.path.exists("results/ProteinMAE_DL/"):
                os.makedirs("results/ProteinMAE_DL/")
            with open("results/ProteinMAE_DL/{}_fold{}_results.csv".format(optn.dataset, cv_idx), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "prediction", "label"])
                writer.writerows(results_per_fold)
        else:
            if not os.path.exists("results/{}_DL/".format(optn.method)):
                os.makedirs("results/{}_DL/".format(optn.method))
            with open("results/{}_DL/{}_fold{}_results.csv".format(optn.method, optn.dataset, cv_idx), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "prediction", "label"])
                writer.writerows(results_per_fold)

        # uncomment to result files for creating plots
        if optn.method == "dmasif":
            optn.method = "dMaSIF"

        if optn.eval_modus == "test_set":
            np.save("results/{}_DL/{}_pos_fold{}.npy".format(optn.method, optn.dataset, cv_idx), pos)
            np.save("results/{}_DL/{}_neg_fold{}.npy".format(optn.method, optn.dataset, cv_idx), neg)
        elif optn.eval_modus == "train_set":
            np.save("results/{}_DL/{}_backPOS_fold{}.npy".format(optn.method, optn.dataset, cv_idx), pos)
            np.save("results/{}_DL/{}_back_fold{}.npy".format(optn.method, optn.dataset, cv_idx), neg)

# %%
