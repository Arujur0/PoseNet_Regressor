import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class ConvBlk(nn.Module):
    def __init__(self, key, in_channels, out_channels, weights, kernel_size, padding):
        super(ConvBlk, self).__init__()
        self.conv = nn.Sequential(init(key, nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding), weights), nn.ReLU())
        #self.bn =  nn.BatchNorm2d(out_channels=out_channels)
    def forward(self, x):
        return self.conv(x)

class InceptionBlock(nn.Module):
    def __init__(self,in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key, weights):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(ConvBlk('inception_'+ key +'/1x1', in_channels, n1x1,  weights, kernel_size = 1, padding=0))

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(ConvBlk('inception_' + key + '/3x3_reduce', in_channels, n3x3red, weights, kernel_size=1, padding = 0), ConvBlk('inception_' + key +'/3x3', n3x3red, n3x3 , weights, kernel_size=3, padding=1))

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(ConvBlk('inception_' + key + '/5x5_reduce', in_channels, n5x5red, weights, kernel_size=1, padding=0), ConvBlk('inception_' + key +'/5x5', n5x5red, n5x5, weights, kernel_size=5, padding=2))

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1, stride=1),nn.ReLU(), ConvBlk('inception_' + key +'/pool_proj', in_channels, pool_planes, weights, kernel_size=1, padding=0))

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        blocks = torch.cat([b1, b2, b3, b4],axis=1)
        return blocks


class LossHeader(nn.Module):
    def __init__(self, k, key, kern, stride, p, inn, out, in_channels, out_channels, last = False, weights=None):
        super(LossHeader, self).__init__()

        self.avg = nn.Sequential(nn.AvgPool2d(kernel_size = 5, stride=3), nn.ReLU())
        self.seq = nn.Sequential(init(key, nn.Conv2d(in_channels=inn, out_channels=out, kernel_size=1, padding=0, stride=1), weights=weights), nn.ReLU(), nn.Flatten())
        self.fc1 = nn.Sequential(init(k, nn.Linear(in_features=in_channels, out_features=out_channels), weights=weights)) # 2048 -> 1024
        self.drop = nn.Dropout(p=p)
        # TODO: Define loss headers
        self.xyz = nn.Sequential(nn.Linear(in_features=1024, out_features=3))
        self.wpqr = nn.Sequential(nn.Linear(in_features=1024, out_features=4))

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.avg(x)
        x = self.seq(x) 
        x = self.fc1(x) 
        xyz = self.xyz(self.drop(x))
        wpqr = self.wpqr(self.drop(x))

        # xyz = self.xyz(self.drop(self.fc1(self.seq(self.avg(x)))))
        # wpqr = self.wpqr(self.drop(self.fc1(self.seq(self.avg(x)))))
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('A2\workspace\pretrained_models\places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers
        self.weights = weights
        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding = 0), weights),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(in_channels= 64, out_channels= 192,kernel_size=3, padding=1, stride = 1),weights),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            nn.ReLU()
        )

        # Example for InceptionBlock initialization
        
        self._3a = nn.Sequential(InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights))
        self.max = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU())
        self._3b = nn.Sequential(InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights))
        self._4a = nn.Sequential(InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights))
        self._4b = nn.Sequential(InceptionBlock(512, 160, 112, 224, 24, 64,  64, "4b", weights))
        self._4c = nn.Sequential(InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights))
        self._4d = nn.Sequential(InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights))
        self._4e = nn.Sequential(InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights))
        self._5a = nn.Sequential(InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights))
        self._5b = nn.Sequential(InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights))
        self.loss1 = LossHeader('loss1/fc', 'loss1/conv', 5, 3, 0.7, 512, 128, 2048, 1024, weights)
        self.loss2 = LossHeader('loss2/fc', 'loss2/conv', 5, 3, 0.7, 528, 128, 2048, 1024, weights)
        self.avg = nn.Sequential(nn.AvgPool2d(kernel_size=7, stride=1), nn.ReLU())
        #self.loss3 = nn.Sequential(nn.Flatten(), nn.Linear(1024,2048), nn.Dropout(0.4))
        self.loss3_wpqr = nn.Sequential(nn.Flatten(), nn.Linear(1024,2048), nn.Dropout(0.4), nn.Linear(2048, 4))
        self.loss3_xyz = nn.Sequential(nn.Flatten(), nn.Linear(1024,2048), nn.Dropout(0.4), nn.Linear(2048, 3))
        #self.model = nn.Sequential(self.pre_layers, self._3a, self._3b, self.max, self._4a, self._4b, self._4c, self._4d, self._4e, self.max, self._5a, self._5b)
        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)
        x = self._4a(self.max(self._3b(self._3a(x))))
        loss1_xyz, loss1_wpqr = self.loss1(x)
        x = self._4b(x)
        x = self._4c(x)
        x = self._4d(x)
        loss2_xyz, loss2_wpqr = self.loss2(x)
        x = self._4e(x)
        x = self.max(x)
        x = self._5a(x)
        x = self._5b(x)
        x = self.avg(x)
        #loss3 = self.loss3(x)
        loss3_xyz, loss3_wpqr = self.loss3_xyz(x), self.loss3_wpqr(x)
        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        unit_loss = torch.divide(poseGT[:, 3:], torch.linalg.vector_norm(poseGT[:, 3:], dim=1, ord=2).unsqueeze(-1))
        losses = 0
        loss1 = torch.linalg.vector_norm((p1_xyz - poseGT[:, 0:3]), dim=1, ord=2) +  self.w1_wpqr * torch.linalg.vector_norm((p1_wpqr - unit_loss), dim=1, ord=2)
        loss2 = torch.linalg.vector_norm((p2_xyz - poseGT[:, 0:3]), dim=1, ord=2) + self.w2_wpqr * torch.linalg.vector_norm((p2_wpqr - unit_loss), dim=1, ord=2)
        loss3 = torch.linalg.vector_norm((p3_xyz - poseGT[:, 0:3]), dim=1, ord=2) + self.w3_wpqr * torch.linalg.vector_norm((p3_wpqr - unit_loss), dim=1, ord=2)
        loss = loss1 * self.w1_xyz + loss2 * self.w2_xyz + loss3 * self.w3_xyz
        for i in loss:
            losses+=i
        losses = losses/ len(loss)
        return losses
