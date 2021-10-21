from torch import nn


class ConvEasy(nn.Module):

    def __init__(self, num_obj):
        super(ConvEasy, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5)),
            nn.MaxPool2d(2),
        )
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(6272, 128),
            nn.ReLU(),
            nn.Linear(128, num_obj * 4)
        )

    def forward(self, x):
        return self.model(x)
