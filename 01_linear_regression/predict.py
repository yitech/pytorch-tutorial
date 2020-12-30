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

df = pd.read_csv(os.path.join(ABS_DIR, config["Predict"]["DataSource"]))
n_sample, n_feature = df.shape

x = torch.from_numpy(df[config["Predict"]["Input"]].to_numpy()).reshape(n_sample, 1)
x = x.type(torch.float)

checkpoint = torch.load(os.path.join(ABS_DIR, config["Predict"]["ModelSource"]))
net = Net()  # setting up model topology
net.load_state_dict(checkpoint)  # load model

# verification
y = torch.from_numpy(df["y"].to_numpy()).reshape(n_sample, 1)
y = y.type(torch.float)
loss_mse = torch.nn.MSELoss()
prediction = net(x)
print(loss_mse(prediction, y))