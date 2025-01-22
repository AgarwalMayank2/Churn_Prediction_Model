import libraries as lb

class NN(lb.nn.Module):
    def __init__(self, input_classes, output_classes):
        super(NN, self).__init__()
        self.fc1 = lb.nn.Linear(input_classes, 32)
        self.fc2 = lb.nn.Linear(32, 4)
        self.fc3 = lb.nn.Linear(4, output_classes)

    def forward(self, x):
        x = lb.F.relu(self.fc1(x))
        x = lb.F.relu(self.fc2(x))
        x = lb.F.sigmoid(self.fc3(x))
        return x