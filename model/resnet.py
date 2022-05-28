import torchvision.models as models
from torch import nn, randn


class ResNet(nn.Module):
    def __init__(self, n_out=100):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.fc = nn.Linear(self.backbone.fc.in_features, n_out)
        self.backbone.fc = self.fc

    def forward(self, x):
        return self.backbone(x)


def test_resnet(in_shape=(2, 3, 32, 32)):
    model = ResNet()
    data = randn(in_shape)
    o = model(data)
    print(o.shape)


if __name__ == "__main__":
    test_resnet()
