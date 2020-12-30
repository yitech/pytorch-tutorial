import os

import argparse
import configparser

import pandas as pd
import torch

from model import Net

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

df = pd.read_csv(os.path.join(ABS_DIR, config["Train"]["DataSource"]))
n_sample, n_feature = df.shape

x = torch.from_numpy(df[config["Train"]["Input"]].to_numpy()).reshape(n_sample, 1)
x = x.type(torch.float)
y = torch.from_numpy(df[config["Train"]["Output"]].to_numpy()).reshape(n_sample, 1).type(torch.float)
y = y.type(torch.float)

net = Net()     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_mse = torch.nn.MSELoss()  # this is for regression mean squared loss

epoch = int(config["Train"]["epoch"])
save_every_epoch = int(config["Train"]["SaveEveryEpoch"])
if save_every_epoch == -1:
    save_every_epoch = epoch + 1
dst_dir = os.path.join(ABS_DIR, config["Train"]["RecordDestination"])


for t in range(epoch):
    prediction = net(x)     # input x and predict based on x
    loss = loss_mse(prediction, y)     # must be (1. nn output, 2. target)
    print("epoch = {}, loss = {}".format(t, loss))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    if t % save_every_epoch == 0:
        torch.save(net.state_dict(), os.path.join(dst_dir, "model" + str(t).zfill(4) + ".pth"))

torch.save(net.state_dict(), os.path.join(dst_dir, "model.pth"))