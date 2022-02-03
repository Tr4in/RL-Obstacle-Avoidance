# Taken from the exmaples from https://pytorch.org/docs/stable/onnx.html, last visited: 26.01.2022

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
import torchvision


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions):
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions
        self.conv_layer1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 10, stride = 8)
        self.conv_layer2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv_layer3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)

        self.fully_connected_layer1 = nn.Linear(8192, 1, dtype = torch.float32)

        self.fully_connected_layer2 = nn.Linear(8192, self.n_actions, dtype = torch.float32)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss(reduction = 'none')
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def calculate_loss(self, eval, target, td_error, isw):
        return isw * td_error * torch.mean((eval - target) ** 2).to(self.device)

    def forward(self, input):
        input = F.layer_norm(input, input.shape[1:])
        conv1_output = F.relu(self.conv_layer1(input))

        output_conv_layer2 = self.conv_layer2(conv1_output)
        conv2_output = F.relu(F.layer_norm(output_conv_layer2, output_conv_layer2.shape[1:]))

        output_conv_layer3 = self.conv_layer3(conv2_output)
        conv3_output = F.relu(F.layer_norm(output_conv_layer3, output_conv_layer3.shape[1:]))


        conv3_output = conv3_output.view(-1, 8192)
		
        v = self.fully_connected_layer1(conv3_output)
        advantage = self.fully_connected_layer2(conv3_output)

        q = v + advantage - torch.mean(advantage)
        return q

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_loss(self):
        return self.loss
    
    def set_loss(self, loss):
        self.loss = loss




checkpoint = torch.load('q_model_episode_1000.pth')
dummy_input = torch.randn(1, 4, 192, 320, device="cuda")
model = DeepQNetwork(0.00005, 6)
model.load_state_dict(checkpoint['main_state_dict'])

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "q_model.onnx", opset_version=11)