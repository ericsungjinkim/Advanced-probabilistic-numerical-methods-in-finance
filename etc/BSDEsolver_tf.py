import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import time

DELTA_CLIP = 50.0

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
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True)

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)

class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        self.model = NonsharedModel(config, bsde)
        self.y_init = self.model.y_init
        # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-2, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config["valid_size"])

        # begin sgd iteration
        for step in range(self.net_config["num_iterations"]+1):
            if step % self.net_config["logging_frequency"] == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])

                print("step ",step," : " ,loss)

            self.train_step(self.bsde.sample(self.net_config["batch_size"]))
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x = inputs
        y_terminal = self.model(inputs, training)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config["y_init_range"][0],
                                                    high=self.net_config["y_init_range"][1],
                                                    size=[1])
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config["dim"]])
                                  )

        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval-1)]

    def call(self, inputs, training):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config["num_time_interval"]) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config["dtype"])
        y = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)

        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)

        return y


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](tf.convert_to_tensor(x,dtype="float32"),training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x


config = {
  "eqn_config": {
    "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
    "eqn_name": "HJBLQ",
    "total_time": 1.0,
    "dim": 100,
    "num_time_interval": 20
  },
  "net_config": {
    "y_init_range": [0, 1],
    "num_hiddens": [110, 110],
    "lr_values": [1e-2, 1e-2],
    "lr_boundaries": [1000],
    "num_iterations": 2000,
    "batch_size": 64,
    "valid_size": 256,
    "logging_frequency": 100,
    "dtype": "float32",
    "verbose": True
  }
}

bsde = HJBLQ(config["eqn_config"])
bsde_solver = BSDESolver(config,bsde)

bsde_solver.train()