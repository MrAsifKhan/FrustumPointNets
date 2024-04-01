import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FrustumPointNetsV1(nn.Module):
    def __init__(self, num_heading_bin=12, num_size_cluster=8):
        super(FrustumPointNetsV1, self).__init__()
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster

        # Define layers for instance segmentation
        self.conv1 = nn.Conv2d(4, 64, kernel_size=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(128, 1024, kernel_size=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=1)

        self.conv7 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv10 = nn.Conv2d(128, 2, kernel_size=1)

        # Define layers for 3D box estimation
        self.conv_reg1 = nn.Conv2d(3, 128, kernel_size=1)
        self.conv_reg2 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv_reg3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv_reg4 = nn.Conv2d(256, 512, kernel_size=1)
        self.fc1 = nn.Linear(515, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + num_heading_bin * 2 + num_size_cluster * 4)

    def forward(self, point_cloud, one_hot_vec, is_training=True):
        batch_size = point_cloud.size(0)
        num_point = point_cloud.size(1)

        # 3D Instance Segmentation PointNet
        net = point_cloud.unsqueeze(2)
        net = F.relu(self.conv1(net))
        net = F.relu(self.conv2(net))
        net = F.relu(self.conv3(net))
        point_feat = F.relu(self.conv4(net))
        net = F.relu(self.conv5(point_feat))
        global_feat = torch.max(net, dim=1, keepdim=True)[0]
        global_feat = torch.cat([global_feat, one_hot_vec.unsqueeze(2).unsqueeze(2).repeat(1, 1, num_point, 1)], dim=3)
        global_feat_expand = global_feat.repeat(1, num_point, 1, 1)
        concat_feat = torch.cat((point_feat, global_feat_expand), dim=3)
        net = F.relu(self.conv6(concat_feat))
        net = F.relu(self.conv7(net))
        net = F.relu(self.conv8(net))
        net = F.relu(self.conv9(net))
        net = F.dropout(net, training=is_training)
        logits = self.conv10(net).squeeze(2)

        # 3D Box Estimation PointNet
        net = point_cloud.unsqueeze(2)
        net = F.relu(self.conv_reg1(net))
        net = F.relu(self.conv_reg2(net))
        net = F.relu(self.conv_reg3(net))
        net = F.relu(self.conv_reg4(net))
        net = torch.max(net, dim=1, keepdim=True)[0].squeeze(2)
        net = torch.cat((net, one_hot_vec), dim=1)
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        output = self.fc3(net)

        return logits, output

if __name__ == '__main__':
    model = FrustumPointNetsV1()
    inputs = torch.zeros((32, 1024, 4))
    one_hot_vec = torch.ones((32, 3))
    logits, output = model(inputs, one_hot_vec)
    print(logits.size(), output.size())  # Check output shapes
