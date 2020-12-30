import torch
import torchvision


def create_head(num_feature, num_class, dropout=0.5, activation=torch.nn.ReLU):
    feature_list = [num_feature, num_feature // 2, num_feature // 4]
    layers = []
    for in_f, out_f in zip(feature_list[:-1], feature_list[1:]):
        layers.append(torch.nn.Linear(in_f, out_f))
        layers.append(activation())
        layers.append(torch.nn.BatchNorm1d(out_f))
        if dropout != 0:
            layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(feature_list[-1], num_class))
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    def __init__(self, num_label, is_freeze=True):
        super(Net, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        if is_freeze:
            self.freeze()
        output = create_head(self.model.fc.in_features, num_label)
        self.model.fc = output

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        return

    def forward(self, x):
        x = self.model(x)
        return x