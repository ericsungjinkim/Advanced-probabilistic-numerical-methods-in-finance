import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

import time

dtype = torch.float32

# We define the Equation Class to generate the input data for training our NN.

class Equation(object):
    '''
    This is the super class of the Equation.
    We have two child class of Equation, HJBLQ and Default_Risk.
    Each class has their own parameter to generate sample for training a Deep NN
    '''
    def __init__(self,eqn_config):
        self.dim = eqn_config["dim"]
        self.total_time = eqn_config["total_time"]
        self.num_time_interval = eqn_config["num_time_interval"]
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None


# ### Hamilton-Jacobi-Bellman (HJB) Equation
# We consider a classical linear-quadratic-Gaussian (LQG) control problem in 100 dimension
class HJB(Equation):
    '''
    We use the setting of the paper J. Han "Solving High-Dimensional Partial
    Differential Equations Using Deep Learning"
    '''
    def __init__(self, eqn_config,lambd):
        super(HJB, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = lambd


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
        # To calculate f term in (5) equation of the article of "solving High-Dimensional Partial Differential Equation Using Deep Learning"
        return -self.lambd * torch.mul(z, z).sum(1)

    def g_tf(self, t, x):
        return torch.log((1 + torch.mul(x, x).sum(1)) / 2)

# ### Nonlinear Black-Scholes Equation with Default Risk

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
            # (5) equation in the article of "solving High-Dimensional Partial Differential Equation Using Deep Learning"
            x_sample[:,:,t+1] = (1+self.mu_bar*self.delta_t)*x_sample[:,:,t]+(self.sigma*x_sample[:,:,t]*dw_sample[:,:,t])

        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        # To calculate f term in (5) equation of the article of "solving High-Dimensional Partial Differential Equation Using Deep Learning"
        m = torch.nn.ReLU()
        piecewise_linear = m(m(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        batch_size = x.size()[0]
        return  x.min(1).values.reshape([batch_size,1])

# ### BSDE Neural Network Architecture
class BSDE_solver(object):
    def __init__(self,config, eqn):
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.eqn = eqn

        self.model = BSDE_NN(config,eqn)
        self.y_init = self.model.y_init
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.net_config["lr_value"],eps=1e-8)

    def train(self):
        print("learning rate: {:.5f}".format(self.net_config["lr_value"]))
        # print("with lambda :{:.2f}".format(self.eqn.lambd))
        start_time = time.time()
        valid_data = self.eqn.sample(self.net_config["valid_size"])

        # We use the same input data for just validation each 100 epochs.
        validation_history = []
        training_history = []
        for step in range(self.net_config["num_iterations"]+1):
            # (8) equation in the article "solving High-Dimensional Partial Differential Equation Using Deep Learning"
            loss = torch.nn.MSELoss()

            if step % 100 == 0 :
                inputs = valid_data
                training = False

            else:
                # During training, we generate the new input data
                inputs = self.eqn.sample(self.net_config["batch_size"])
                training = True

            batch_size = inputs[0].shape[0]
            y_terminal_model = self.model(inputs,training)

            y_terminal_target = self.eqn.g_tf(self.eqn.total_time, torch.tensor(inputs[1][:,:,-1], dtype=dtype)).reshape([batch_size,1])
            loss_val = loss(y_terminal_model,y_terminal_target)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            err_rate = (loss_val.item() / y_terminal_target.mean()).item()
            y_init = self.model.y_init.data.item()

            if step % 100 == 0:
                training_time = time.time() - start_time
                validation_history.append([step, err_rate, y_init,training_time])
                print("step: {},  err_rate: {:.5f},  Y0: {:.5f}, Traing_time: {:.5f}".format(step,err_rate,y_init,training_time))
                training_history.append([step, err_rate, y_init])



        return np.array(training_history), training_time

# We set our model. Network is same as in the article. We train all the parameters of NN and initial value of $u(0,X_0)$
# and $Z_0 = \sigma^T(t_0,X_0)\nabla u(t_0,X_{t_0})$. We set the initial value as a parameters for training
# Each sub network calculate $\sigma^T(t_n,X_{t_n})\nabla u(t_n,X_{t_n})$.
# We set $z = \sigma^T(t_n,X_{t_n})\nabla u(t_n,X_{t_n})$

class BSDE_NN(torch.nn.Module):
    def __init__(self,config, eqn):
        super(BSDE_NN,self).__init__()
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.eqn = eqn

        # Set all initial values as a parameter for training
        y_init = torch.tensor(np.random.uniform(low=self.net_config["y_init_range"][0],
                                                high=self.net_config["y_init_range"][1],
                                                size=[1]),
                              dtype=dtype, requires_grad=True)
        self.y_init = torch.nn.Parameter(y_init)

        z_init = torch.tensor(np.random.uniform(low=-.1,high=.1,
                                                size=[1,self.eqn_config["dim"]]),
                              dtype=dtype, requires_grad=True)
        self.z_init = torch.nn.Parameter(z_init)

        # Set the sub-network for calculating the gradient of u of each time.
        # Each sub-network is multilayer feedforward neural network approximation the product of sigma * gradients of u at time t = t_n
        self.subnet = torch.nn.ModuleList(
            [Feedforward_NN(config) for _ in range(self.eqn_config["num_time_interval"]-1)])

    def forward(self, inputs, training):
        batch_size = inputs[0].shape[0]
        dw = torch.tensor(inputs[0],dtype=dtype)
        x = torch.tensor(inputs[1],dtype=dtype)

        time_stamp = np.arange(0,self.eqn_config["num_time_interval"]) * self.eqn.delta_t
        all_one_vector = torch.ones(batch_size,1,dtype=dtype)

        # y is the value of u(t,x)
        # the first setting of y is u(0,x_0). We fix the x_0 from equation class
        y = all_one_vector * self.y_init
        z = all_one_vector.mm(self.z_init)

        for t in range(0, self.eqn.num_time_interval -1):
            # (5) equation of the article of "solving High-Dimensional Partial Differential Equation Using Deep Learning"
            y = y - (self.eqn.delta_t * (self.eqn.f_tf(time_stamp[t], x[:, :, t], y, z).reshape([batch_size,1])) + \
                     (torch.mm(z, dw[:, :, t].t()).sum(1)).reshape([batch_size,1])).reshape([batch_size,1])
            # Calculate the product of sigma * gradients of u at time t = t_n
            z = self.subnet[t](x[:, :, t + 1], training) / self.eqn.dim

        # Terminal u(t_N,x)
        y = y - (self.eqn.delta_t * (self.eqn.f_tf(time_stamp[t], x[:, :, t], y, z).reshape([batch_size,1])) + \
                     (torch.mm(z, dw[:, :, t].t()).sum(1)).reshape([batch_size,1])).reshape([batch_size,1])
        y = y.reshape([batch_size,1])
        return y


class Feedforward_NN(torch.nn.Module):
    '''
    This is the NN for the feedforward to calculate the gradient of u at t = t_n
    '''
    def __init__(self,config):
        super(Feedforward_NN, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]

        # batchnormal layer for the input data
        self.batchnormal_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim, momentum=0.99, eps=1e-6, affine=True)])
        for i in num_hiddens:
            # batchnormal layer for the each hidden layer
            self.batchnormal_layers.append(torch.nn.BatchNorm1d(i, momentum=0.99, eps=1e-6, affine=True))
        # batchnormal layer for the output data
        self.batchnormal_layers.append(torch.nn.BatchNorm1d(dim, momentum=0.99, eps=1e-6, affine=True))

        # Linear layer from input to first hidden layer
        self.dense_layers = torch.nn.ModuleList([torch.nn.Linear(dim, num_hiddens[0], bias=False)])
        for i in range(len(num_hiddens) - 1):
            # Linear layer from i hidden layer to i+1 hidden layer
            self.dense_layers.append(torch.nn.Linear(num_hiddens[i], num_hiddens[i + 1], bias=False))
        # Linear layer from last hidden layer to output
        self.dense_layers.append(torch.nn.Linear(num_hiddens[-1], dim, bias=False))

    def forward(self,x,training):
        x = self.batchnormal_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.batchnormal_layers[i + 1](x)
            x = F.relu(x)
        x = self.dense_layers[-1](x)
        x = self.batchnormal_layers[-1](x)
        return x



HJB_config = {
  "eqn_config": {
    "total_time": 1.0,
    "dim": 100,
    "num_time_interval": 20,
  },
  "net_config": {
    "y_init_range": [0, 1],
    "num_hiddens": [110, 110],
    "lr_value": 5*1e-3,
    "num_iterations": 2000,
    "batch_size": 64,
    "valid_size": 256,
  }
}



Default_Risk_config = {
  "eqn_config": {
    "total_time": 1.0,
    "dim": 100,
    "num_time_interval": 40
  },
  "net_config": {
    "y_init_range": [40, 50],
    "num_hiddens": [110, 110],
    "lr_value": 8e-3,
    "num_iterations": 4000,
    "batch_size": 64,
    "valid_size": 256,
  }
}


def df_save(eqn,config,runs,name):
    print("training ",name)
    history_log_all = []
    times_all = []
    for t in range(runs):
        print("Run time {}".format(t+1))
        solver = BSDE_solver(config,eqn)
        history_log, training_time = solver.train()

        del solver
        history_log_all.append(history_log)
        times_all.append(training_time)
    # runtimes /
    history_log_all = np.array(history_log_all)
    err_rate_mean = []
    err_rate_std = []
    y0_mean = []
    y0_std = []
    for step in range(history_log_all.shape[1]):
        err_rate_mean.append(history_log_all[:,step,1].mean())
        err_rate_std.append(history_log_all[:,step,1].std())

        y0_mean.append(history_log_all[:, step, 2].mean())
        y0_std.append(history_log_all[:, step, 2].std())

    col = ['step', 'err_rate', 'y0']
    for t in range(runs):
        df = pd.DataFrame(history_log_all[t], columns=col)
        name_df = name + "_"+ str(t) + ".csv"
        path = './log/' + name_df
        df.to_csv(path)

    name_df_times = name+"_time" +".csv"
    path = './log/'+name_df_times
    df = pd.DataFrame(times_all,columns=['times'])
    df.to_csv(path)

    col = ['err_rate_mean','err_rate_std','y0_mean','y0_std']
    idx = history_log_all[0,:,0]
    df = pd.DataFrame(columns=col,index=idx)
    df.iloc[:,0] = np.array(err_rate_mean)
    df.iloc[:,1] = np.array(err_rate_std)
    df.iloc[:,2] = np.array(y0_mean)
    df.iloc[:,3] = np.array(y0_std)
    name_df = 'result_'+name
    path = "./log/" + name_df + ".csv"
    df.to_csv(path)



    x = np.array(idx)
    plt.figure()
    plt.plot(x, df['err_rate_mean'])
    plt.yscale('log')
    title = name + 'Equation : err_rate for 5 different runs'
    plt.title(title)
    plt.ylabel('err_rate')
    plt.xlabel('Number of iteration steps')
    plt.grid()
    plt.legend()
    plt.show()

    err_upper = np.array(df['y0_mean']) + np.array(df['y0_std'])
    err_lower = np.array(df['y0_mean']) - np.array(df['y0_std'])
    plt.figure()
    plt.plot(x, df['y0_mean'],'r')
    plt.fill_between(x, err_lower, err_upper, label='1 sigma range', color='gray', alpha=0.2)
    title = name + 'Equation : Y0 for 5 different runs'
    plt.title(title)
    plt.ylabel('Y0')
    plt.xlabel('Number of iteration steps')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()
    y0_fin = np.array(df['y0_mean'])[-1]

    return df, y0_fin



lambd = 1.0
print("We train the HJB equation with lambda {:.2f}".format(lambd))
HJB_eqn = HJB(HJB_config["eqn_config"],lambd)
df_HJB_1, y0_HJB_1 = df_save(HJB_eqn,HJB_config,5,"HJB_lambd_1")
del HJB_eqn

print("We train the Nonlinear Black-Scholes Equation with Default Risk. ")
nonlinear_BS_eqn = Default_Risk(Default_Risk_config["eqn_config"])
name = "Nonlinear_BS"
df_nonlinear_BS, y0_Nonlinear_BS = df_save(nonlinear_BS_eqn,Default_Risk_config,5,name)

# Simulate with different lambda
df_lambd = []
y0_HJB_s = []
for lambd in np.arange(0,60,10):
    print("We train the HJB equation with lambda {:.2f}".format(lambd))
    HJB_eqn = HJB(HJB_config["eqn_config"], lambd)
    name = "HJB_lambd_" + str(lambd)
    df , y0_HJB = df_save(HJB_eqn, HJB_config, 5, name)

    del HJB_eqn
    df_lambd.append(df)
    y0_HJB_s.append(y0_HJB)

x = np.arange(0,60,10)
plt.figure()
plt.plot(x,y0_HJB_s,'r')
title = 'Y0 = u(0,(0,...,0))'
plt.title(title)
plt.ylabel('Y0')
plt.xlabel('lambda')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

