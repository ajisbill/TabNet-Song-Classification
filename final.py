from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from torch import nn, argmax
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.decomposition import PCA
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import sys
import argparse

class SongDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = X
        self.targets = y
    def __getitem__(self, idx):
        x = self.inputs[idx,:]
        y = self.targets[idx]
        return x,y
    def __len__(self):
        return self.inputs.shape[0]
    def D(self):
        D = self.inputs.shape[1]
        return D

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(90, 3)

    def forward(self, x):
        out = self.linear(x)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_dim,256)
        self.linear2 = nn.Linear(256,128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 3)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.output(x)
        
        return x
        
def train(model,train_loader, val_loader,X_train, y_train, X_dev, y_dev,  lr, epochs, report_freq, device):

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_train, X_dev= torch.from_numpy(X_train), torch.from_numpy(X_dev)
    X_train = X_train.to(device)
    X_dev = X_dev.to(device)
    for epoch in range(epochs):
        print("Epoch:", epoch, flush=True)
        for update,(mb_x,mb_y) in enumerate(train_loader):
            mb_x = mb_x.to(device)
            mb_y = mb_y.to(device)

            preds = model(mb_x)
            # breakpoint()
            loss = criterion(preds, mb_y.long())
    
            # take gradient step
            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

        
            if(update%report_freq == 0):
                num = 0
                den = 0
                num1 = 0
                den1 = 0

                ##### 
                dev_predictions = argmax(torch.nn.functional.softmax(model(X_dev), dim=1), axis=1)
                dev_acc = balanced_accuracy_score(y_dev, dev_predictions.cpu())
                print("Balanced Dev Accuracy: {}".format(dev_acc), flush=True)

                train_predictions = argmax(torch.nn.functional.softmax(model(X_train), dim=1), axis=1)
                train_acc = balanced_accuracy_score(y_train, train_predictions.cpu())
                print("Balanced Train Accuracy: {}".format(train_acc), flush=True)

                # for samples,(mb_xd,mb_yd) in enumerate(val_loader):
                #     mb_xd = mb_xd.to(device)
                #     mb_yd = mb_yd.to(device)

                #     dev_predictions = argmax(model(mb_xd), axis=1)
                #     num += sum(dev_predictions == mb_yd)
                #     den += mb_yd.shape[0]
                    

                # for samples,(mb_xd,mb_yd) in enumerate(train_loader):
                #     mb_xd = mb_xd.to(device)
                #     mb_yd = mb_yd.to(device)

                #     train_predictions = argmax(model(mb_xd), axis=1)
                #     num1 += sum(train_predictions == mb_yd)
                #     den1 += mb_yd.shape[0]
                    

                
                # dev_accuracy = num/den
                # train_accuracy = num1/den1
                # print("Dev Accuracy: {}".format(dev_accuracy), flush=True)
                # print("Train Accuracy: {}".format(train_accuracy), flush=True)

def tabNet(X_train, y_train, X_val, y_val, X_test):
    clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=.02),
                       scheduler_params={"step_size":20, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"
                      )

# fit the model 
    clf.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'val'],
        eval_metric=['accuracy', 'balanced_accuracy'],
        max_epochs=200 , patience=20,
        batch_size=8192, virtual_batch_size=256,
        num_workers=0,
        weights=1,
        drop_last=False
    )

    save_path = clf.save_model('TabNet')

    val_preds = clf.predict(X_val)
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    val_acc = balanced_accuracy_score(y_val, val_preds)
    train_acc = balanced_accuracy_score(y_train, train_preds)

    print("Final Balanced Val Accuracy: {}".format(val_acc), flush=True)
    print("Final Balanced Train Accuracy: {}".format(train_acc), flush=True)

    return test_preds, val_acc

def main(argv):
    # load data
    args = parse_all_args()
    data = loadmat('data.mat')
    batch_size = args.mb
    lr = args.lr
    epochs = args.epochs
    report_freq = args.report
    device = 0

    # get X_test, X_train, and y_train np arrays
    X_test = data['X_test']
    X_train = data['X_train']
    y_train = data['y_train']
    y_train = y_train.transpose()

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.reshape(-1)
    X_train, X_test = scaleData(X_train, X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    train_dataset = SongDataset(X_train, y_train)
    val_dataset = SongDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size= 92743, shuffle=False)

    model = LogisticRegressionModel(90, 3)
    model = model.to(device)

    # train(model, train_loader, val_loader, X_train, y_train, X_val, y_val, lr, epochs, report_freq, device)
    preds, val_acc = tabNet(X_train, y_train, X_val, y_val, X_test)
    test_err_pred = np.array([1- val_acc])
    np.savetxt("y_pred.gz", preds, delimiter=",", fmt="%d")
    np.savetxt("err_pred.txt", test_err_pred)


def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.0001]",default=0.0001)

    parser.add_argument('-mb', type=int,\
            help="The minibatch size (int) [default: 32]", \
            default = 32)

    parser.add_argument('-report_freq', type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]", \
            default = 128, dest='report', metavar='REPORT_FREQ')

    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 10]",\
            default=10)
    
    return parser.parse_args()


def scaleData(x_tr, x_test):
    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_test = scaler.transform(x_test)
    
    return x_tr, x_test


if __name__ == '__main__':
    main(sys.argv)
