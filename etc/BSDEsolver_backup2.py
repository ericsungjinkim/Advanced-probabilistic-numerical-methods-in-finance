import torch
import torch.nn.functional as F


import numpy as np
from scipy.stats import multivariate_normal as normal

dtype = torch.float32

class Equation(object):
    def __init__(self,eqn_config):
        self.dim = eqn_config["dim"]
        self.total_time = eqn_config["total_time"]
        self.num_time_interval = eqn_config["num_time_interval"]
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

class HJBLQ(Equation):
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * torch.mul(z, z).sum(1)

    def g_tf(self, t, x):
        return torch.log((1 + torch.mul(x, x).sum(1)) / 2)

class Default_Risk(Equation):
    '''
    We use the setting of the paper J. Han "Solving High-Dimensional Partial
    Differential Equations Using Deep Learning"
    '''
    def __init__(self, eqn_config):
        super(Default_Risk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,self.dim,self.num_time_interval])*self.sqrt_delta_t
        x_sample = np.zeros([num_sample,self.dim,self.num_time_interval+1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init

        for t in range(self.num_time_interval):
            x_sample[:,:,t+1] = (1+self.mu_bar*self.delta_t)*x_sample[:,:,t]+(self.sigma*x_sample[:,:,t]*dw_sample[:,:,t])

        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        m = torch.nn.ReLU()
        piecewise_linear = m(m(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        batch_size = x.size()[0]
        return  x.min(1).values.reshape([batch_size,1])


class BSDESolver(object):
    def __init__(self,config,bsde):
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        self.model = NonsharedModel(config,bsde)
        self.y_init = self.model.y_init
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=5*1e-3, eps=1e-8)

    def train(self):

        training_history = []
        err_rates = []
        y_inits = []
        valid_data = self.bsde.sample(self.net_config["valid_size"])

        for step in range(self.net_config["num_iterations"]+1):
            loss = torch.nn.MSELoss()

            if step%self.net_config["logging_frequency"] == 0:
                inputs = valid_data
                training = False
            else:
                inputs = self.bsde.sample(self.net_config["batch_size"])
                training = True

            batch_size = inputs[0].shape[0]
            y_terminal_model = self.model(inputs,training)


            y_terminal_real = self.bsde.g_tf(self.bsde.total_time, torch.tensor(inputs[1][:, :, -1],dtype=dtype)).reshape([batch_size,1])
            loss_val = loss(y_terminal_model,y_terminal_real)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            err_rate = (loss_val.item() / y_terminal_real.mean()).item()
            err_rates.append(err_rate)
            y_init = self.y_init
            y_inits.append(y_init)

            if step % 100 == 0:
                training_history.append([step, err_rate, self.y_init.item()])
                print("step ",step," err_rate : " ,err_rate, ", Y0 : ",y_init)


        return np.array(err_rates), np.array(training_history) , y_inits




class NonsharedModel(torch.nn.Module):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        self.y_init = torch.tensor(np.random.uniform(low=self.net_config["y_init_range"][0],
                                                     high=self.net_config["y_init_range"][1],
                                                     size=[1]),
                                   dtype=dtype,requires_grad=True)

        self.z_init = torch.tensor(np.random.uniform(low=-.1, high=.1,size=[1, self.eqn_config["dim"]]),
                                   dtype=dtype,requires_grad=True)


        self.subnet = torch.nn.ModuleList(
            [FeedForwardSubNet(config) for _ in range(self.eqn_config["num_time_interval"] - 1)])

    def forward(self, inputs, training):
        batch_size = inputs[0].shape[0]
        dw = torch.tensor(inputs[0], dtype=dtype)
        x = torch.tensor(inputs[1], dtype=dtype)
        time_stamp = np.arange(0, self.eqn_config["num_time_interval"]) * self.bsde.delta_t
        all_one_vec = torch.ones(batch_size, 1, dtype=dtype)
        y = all_one_vec * self.y_init
        print(self.y_init)
        z = all_one_vec.mm(self.z_init)

        for t in range(0, self.bsde.num_time_interval - 1):
            y = y - (self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z).reshape([batch_size,1])) + \
                     (torch.mm(z, dw[:, :, t].t()).sum(1)).reshape([batch_size,1])).reshape([batch_size,1])
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim


        # terminal time
        y = y - (self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z)).reshape([batch_size,1]) + \
            (z * dw[:, :, -1]).sum(1).reshape([batch_size,1])
        y = y.reshape([batch_size,1])
        return y


class FeedForwardSubNet(torch.nn.Module):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]

        self.bn_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim, momentum=0.99, eps=1e-6, affine=True)])
        for i in num_hiddens:
            self.bn_layers.append(torch.nn.BatchNorm1d(i, momentum=0.99, eps=1e-6, affine=True))
        self.bn_layers.append(torch.nn.BatchNorm1d(dim, momentum=0.99, eps=1e-6, affine=True))

        self.dense_layers = torch.nn.ModuleList([torch.nn.Linear(dim, num_hiddens[0], bias=False)])
        for i in range(len(num_hiddens) - 1):
            self.dense_layers.append(torch.nn.Linear(num_hiddens[i], num_hiddens[i + 1], bias=False))
        self.dense_layers.append(torch.nn.Linear(num_hiddens[-1], dim, bias=False))

    def forward(self, x, training):
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x)
            x = F.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        return x

HJB_config = {
  "eqn_config": {
    "total_time": 1.0,
    "dim": 100,
    "num_time_interval": 20
  },
  "net_config": {
    "y_init_range": [4, 5],
    "num_hiddens": [110, 110],
    "lr_values": [1e-2, 1e-2],
    "lr_boundaries": [1000],
    "num_iterations": 4000,
    "batch_size": 64,
    "valid_size": 256,
    "logging_frequency": 100,
    "dtype": "float64",
    "verbose": True
  }
}

HJB_bsde = HJBLQ(HJB_config["eqn_config"])
HJB_bsde_solver = BSDESolver(HJB_config,HJB_bsde)


Default_Risk_config = {
  "eqn_config": {
    "total_time": 1.0,
    "dim": 100,
    "num_time_interval": 40
  },
  "net_config": {
    "y_init_range": [40, 50],
    "num_hiddens": [110, 110],
    "lr_values": [8e-3, 8e-3],
    "lr_boundaries": [3000],
    "num_iterations": 6000,
    "batch_size": 64,
    "valid_size": 256,
    "logging_frequency": 100,
    "dtype": "float64",
    "verbose": True
  }
}
Default_Risk_bsde = Default_Risk(Default_Risk_config["eqn_config"])
Default_Risk_bsde_solver = BSDESolver(Default_Risk_config,Default_Risk_bsde)

a,b,c = Default_Risk_bsde_solver.train()