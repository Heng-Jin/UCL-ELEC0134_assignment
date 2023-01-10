import csv
import numpy as np
import logging
import pathlib
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset,DataLoader
from torchvision import models
from PIL import Image
from matplotlib import pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

Loss_list = []
Accuracy_train_list = []
Accuracy_test_list = []

train_datapath = pathlib.Path.cwd().parents[0]/'Datasets'/'dataset_AMLS_22-23'/'celeba'
test_datapath = pathlib.Path.cwd().parents[0]/'Datasets'/'dataset_AMLS_22-23_test'/'celeba_test'
task='smiling'
model_save_path = pathlib.Path.cwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

log_path = pathlib.Path.cwd() / ("train_test_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

class amls_dataset(Dataset):
    def __init__(self, path, task, mode):
        img_path = path / 'img'
        data_root = pathlib.Path(img_path)
        self.all_image_paths = list(data_root.glob('*.jpg'))

        # load labels
        label_path = path / 'labels.csv'
        label_list = self.load_label(label_path, mode)
        if task == 'gender':
            label_dict = dict((temp[1], temp[2]) for temp in label_list)
        elif task == 'smiling':
            label_dict = dict((temp[1], temp[3]) for temp in label_list)
        else:
            logging.warning('-----No such task-----')
            print('-----No such task-----')

        # ground truth amount check
        if len(label_dict) != len(self.all_image_paths):
            logging.warning('-----label amount dismatch with img amount-----')
            print('-----label amount dismatch with img amount-----')

        # corresponding label to img
        self.all_image_labels = list()
        for i in self.all_image_paths:
            if label_dict.get(str(i.name)) is not None:
                self.all_image_labels.append(float(label_dict[str(i.name)]))
            else:
                logging.warning('-----no label imgs-----')
                print('-----no label imgs-----')
                print(i)

        # image normalization params
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))

    def load_label(self, path, mode):
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = []
            for i, row in enumerate(reader):
                if i == 0:
                    dataset_title = row[0]
                    continue
                row = row[0].split("\t")
                row = ['0' if i == '-1' else i for i in row]
                rows.append(row)
        logging.info('-----load %s dataset labels-----', mode)
        print('-----load ', mode, ' dataset labels-----')
        label_data = np.array(rows)

        return label_data

    def __getitem__(self, index):
        img = Image.open(self.all_image_paths[index])
        img = np.array(img.resize((224, 224)))
        img = img / 255.
        img = (img - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        label = self.all_image_labels[index]
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)

train_dataset = amls_dataset(train_datapath, task, "training")
test_dataset = amls_dataset(test_datapath, task, "test")

batch_size = 16
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size)

def train(net, train_iter, test_iter, criterion, optimizer, num_epochs):
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)
    whole_batch_count = 0
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y = y.to(torch.long)
            optimizer.zero_grad()
            y_hat = net(X)
            # print(y_hat.type(),y.type())
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            whole_batch_count += 1
            batch_count += 1
            temp_loss = train_loss_sum / whole_batch_count
            temp_acc_train = train_acc_sum / n
            Loss_list.append(loss.item())
            Accuracy_train_list.append(temp_acc_train)
            logging.info('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, train acc temp %.3f, time %.1f sec,'
                  % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))
            print('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, train acc temp %.3f, time %.1f sec'
                  % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))

        with torch.no_grad():
            net.eval()  # evaluate mode
            test_acc_sum, n2 = 0.0, 0
            test_result_list=[]
            for X, y in test_iter:
                y_hat = net(X.to(device))
                test_acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                temp = torch.stack((y_hat.argmax(dim=1).int(), y.to(device).int(), y_hat.argmax(dim=1) == y.to(device)), 1).tolist()
                test_result_list.extend(temp)
                n2 += y.shape[0]

        temp_acc_test = test_acc_sum / n2
        Accuracy_test_list.append(temp_acc_test)
        logging.info('---epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec---'
              % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_test, time.time() - start))
        print('---epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec---'
              % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_test, time.time() - start))

        result_path = pathlib.Path.cwd() / (task + "_epoch_" + str(epoch) + "_lr_" + str(lr) +"_test_result.csv")
        create_csv(result_path, test_result_list)

def create_csv(path, result_list):
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["predict_label", "gt_label", "match"])
        csv_write.writerows(result_list)

def plot_save(loss_list, acc_list):
    x1 = range(len(acc_list))
    x2 = range(len(loss_list))
    y1 = acc_list
    y2 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Training loss vs. iteration')
    plt.ylabel('Test loss')
    # plt.show()
    plt.savefig((task + "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".jpg"))

pretrained_net = models.resnet18(pretrained=True)
num_ftrs = pretrained_net.fc.in_features
pretrained_net.fc = nn.Linear(num_ftrs, 2)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
epoch = 5
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)

loss = torch.nn.CrossEntropyLoss()
train(pretrained_net, train_iter, test_iter, loss, optimizer, num_epochs=epoch)

plot_save(Loss_list, Accuracy_test_list)

torch.save(pretrained_net.state_dict(),
           model_save_path / (task + "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) +".pth"))