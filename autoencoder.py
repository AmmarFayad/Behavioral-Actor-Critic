from torch import nn
import numpy as np
import torch
import torch.nn
import torch.optim as optim
l=torch.nn.MSELoss()
class autoencoder(nn.Module):
    def random_weight(self,shape):
        """
        Kaiming normalization: sqrt(2 / fan_in)
        """
        if len(shape) == 2:  # FC weight
            fan_in = shape[0]
        else:
            fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    
        w = torch.randn(shape) * np.sqrt(2. / fan_in)
        w.requires_grad = True
        return w
    def weights_init(self,m):
        if type(m) in [nn.Conv2d, nn.Linear]:
            m.weight.data = self.random_weight(m.weight.data.size())
            m.bias.data = self.random_weight(m.bias.data.size())
    def __init__(self,sd):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sd, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.encoder.apply(self.weights_init)
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, sd), nn.Tanh())
        self.decoder.apply(self.weights_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def update_autoencoder (self, model, net, parameters, env, select_action):
        state = env.reset()
        state = torch.Tensor(state).unsqueeze(0)
        if parameters.is_cuda: state = state.cuda()
        done = False
        while not done:
            action=select_action(net,state)

            next_state, reward, done, info = env.step(action.flatten())  #Simulate one step in environment
            next_state = torch.Tensor(next_state).unsqueeze(0)
            if parameters.is_cuda:
                next_state = next_state.cuda()
            s=model(state)
            f=l(state,s)
            c=optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001) 
            c.zero_grad()
            f.backward(retain_graph=True)
            c.step()
            state = next_state