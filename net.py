import torch
import torch.nn as nn


class ActionNet(nn.Module):
    def __init__(self,
                 input=(30 * 30 + 30 * 30 + 1),
                 output=30,
                 noise=False):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output)

        self.relu = nn.ReLU

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

        self.sfm = nn.Softmax(output)

        self.noise = None
        if noise:
            self.noise = torch.rand
        self.noise_strength = 0.3

    def forward(self, x):
        output = x
        output = self.bn1(self.relu(self.fc1(output)))
        output = self.bn2(self.relu(self.fc2(output)))
        output = self.fc3(output)
        if not self.noise is None:
            n = self.noise(*output.shape) * self.noise_strength
            output = output + n
        # 连接关系掩码
        cm = x[:, :900]
        cm = cm.view(-1, 30, 30)
        lp = x[:, 1800].long()
        mask = [cm[i, lp[i]].unsqueeze(0) for i in range(cm.shape[0])]
        mask = torch.cat(tuple(mask), dim=0)
        output = output * mask

        output = self.sfm(output)
        return output


class ActorNetLoss(nn.Module):
    def __init__(self):
        super(ActorNetLoss, self).__init__()

    def forward(self, critic_q, batch_size):
        total_loss = torch.sum(critic_q) / batch_size
        return total_loss


class CriticNet(nn.Module):
    def __init__(self,
                 input=(30 * 30 + 30 * 30 + 1 + 30),
                 output=1):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output)

        self.relu = nn.ReLU

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.bn1(self.relu(self.fc1(x)))
        x = self.bn2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CriticNetLoss(nn.Module):
    def __init__(self):
        super(CriticNetLoss, self).__init__()
        self.loss = nn.MSELoss

    def forward(self, pred, target):
        total_loss = self.loss(pred, target)
        return total_loss
