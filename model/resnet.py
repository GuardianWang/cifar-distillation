import torchvision.models as models
from torch import nn, randn
import torch


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, cfg.classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_resnet(in_shape=(2, 3, 32, 32)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    print(model)
    model = model.train()
    data = randn(in_shape).to(device)
    o = model(data)
    print(o.shape)
    model = model.eval()
    o = model(data)
    print(o.shape)


if __name__ == "__main__":
    test_resnet()
