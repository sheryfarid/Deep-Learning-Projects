import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: cat, dog, other
        )

    def forward(self, x):
        return self.net(x)
