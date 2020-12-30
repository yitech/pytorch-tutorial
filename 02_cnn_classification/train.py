import os

import argparse
import configparser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import MapDataset
from model import Net


from PIL import Image

ABS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./config.ini", help="source of config file")
arg = parser.parse_args()

if not os.path.exists(arg.config):
    print(Exception("FileDoNotExist: {}".format(arg.config)))
    exit()

config = configparser.ConfigParser()
config.read(arg.config)
for section in config.sections():
    print("[{}]".format(section))
    for key, value in config[section].items():
        print("{} = {}".format(key, value))

# Load data
batch_size = 32
transform = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

map_dataset = MapDataset(csv_file=os.path.join(ABS_DIR, config["Train"]["LabelSource"])
                         , img_dir=os.path.join(ABS_DIR, config["Train"]["ImageSource"])
                         , trans=transform)

n_valid = int(len(map_dataset) * 0.12)
train_set, val_set = random_split(map_dataset, [len(map_dataset) - n_valid, n_valid])
print("training set: {}, validation set: {}".format(len(train_set), len(val_set)))
data_loader = {"train": DataLoader(train_set, shuffle=True, batch_size=batch_size),
               "val": DataLoader(val_set, shuffle=True, batch_size=batch_size)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device on {}".format(device))
net = Net(map_dataset.num_class)
net = net.to(device)
print(net)
# img = torch.randn(64 * 64 * 3 * 2).reshape(2, 3, 64, 64)
# print(net(img))
loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005)

# Training
n_epoch = int(config["Train"]["epoch"])
save_every_epoch = int(config["Train"]["SaveEveryEpoch"])
if save_every_epoch == -1:
    save_every_epoch = n_epoch + 1
dst_dir = os.path.join(ABS_DIR, config["Train"]["RecordDestination"])

for t in range(n_epoch):
    result = []
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()
        # keep track of training and validation loss
        running_loss = 0.0
        running_correct = 0.0

        for data, target in data_loader[phase]:
            data, target = data.to(device), target.to(device)
            print(target.shape)
            with torch.set_grad_enabled(phase == 'train'):
                # feed the input
                preds = net(data)
                print(preds.shape)
                # calculate the loss
                loss = loss_func(preds, target)
                # preds[preds == torch.max(preds, axis=1)] = 1.0
                # preds[preds < 1.0] = 0.0
                # preds = preds.to(torch.float32)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(data_loader[phase].dataset)
        epoch_acc = running_correct / len(data_loader[phase].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))




# print(type(torchvision.models.resnet50(pretrained=True)))
