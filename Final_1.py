import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import gen_batches, check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit as logistic_sigmoid
from sklearn.neural_network import MLPRegressor

def inplace_relu(X):
    np.maximum(X, 0, out=X)

def inplace_logistic(X):
    logistic_sigmoid(X, out=X)

def squared_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() / 2

def inplace_relu_derivative(Z, delta):
    delta[Z == 0] = 0

def discretesample(p):
    edges = np.append(0, np.cumsum(p, axis = 0))
    rv = np.random.rand()
    c, c0 = np.histogram(rv, edges)
    xv = np.nonzero(c)
    x = xv[0]
    return x

ACT = {'relu': inplace_relu,
                'logistic':inplace_logistic}

DER = {'relu': inplace_relu_derivative}

class Adam():
    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        self.params = [param for param in params]
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates

    def update_params(self, grads):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def trigger_stopping(self, msg, verbose):
        if verbose:
            print(msg + " Stopping.")
        return True

class IS_Adam():

    def __init__(self, hidden_size=(100,), alpha=0.001, batch_size=1000, 
                learning_rate=0.0001, iter_max=6000, tol=1e-6, verbose=True, 
                beta_1=0.9,beta_2=0.999, epsilon=1e-8, n_iter_no_change=40):
        
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iter_max = iter_max
        self.tol = tol
        self.verbose = verbose
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

    def _init_coef(self, fan_in, fan_out, dtype):

        factor = 6.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init

    def _initialize(self, y, layer_units, dtype):

        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        self.n_layers_ = len(layer_units)

        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1],
                                                        dtype)
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self.loss_p = []
        self._no_improvement_count = 0
        self.best_loss_ = np.inf

    def _forward_pass(self, activations):

        hidden_activation = ACT['relu']
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                 self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACT['logistic']
        output_activation(activations[i + 1])

        return activations

    def _forward_pass_fast(self, X):
        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACT['relu']
        for i in range(self.n_layers_ - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACT['logistic']
        output_activation(activation)

        return activation

    def _update_no_improvement_count(self):

        if self.loss_p[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
        else:
                self._no_improvement_count = 0
        if self.loss_p[-1] < self.best_loss_:
                self.best_loss_ = self.loss_p[-1]

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):

        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):

        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss = squared_loss(y, activations[-1])
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        inplace_derivative = DER['relu']
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads

    def get_P (self, X):
        length = len(X)
        lambda_ = 0.01
        l2 = np.zeros(length)
        P_new = np.zeros(length)

        for i in range(length):
            l2[i] = np.linalg.norm(data[i]) + lambda_ ** 0.5
            P_new = l2 / sum(l2)
        
        return P_new

    def get_IS_sample(self, X, y, P, length = 8000):
        width_x = np.size(X[0])
        width_y = np.size(y[0])
        x_IS = np.zeros((length, width_x))
        y_IS = np.zeros((length, width_y))
        for ii in range(length):
            a = discretesample(P)
            x_IS[ii] = X[a, :]
            y_IS[ii] = y[a, :]
        return x_IS, y_IS

    def fit(self, X, y):

        hidden_layer_sizes = list(self.hidden_size)

        n_samples, n_features = X.shape

        if y.ndim == 1:
            y = y.reshape((-1, 1))    
            
        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +[self.n_outputs_])
        
        self._random_state = check_random_state(None)

        self._initialize(y, layer_units, X.dtype)

        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
                      for n_fan_in_, n_fan_out_ 
                        in zip(layer_units[:-1],layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_, dtype=X.dtype)
                           for n_fan_out_ in layer_units[1:]]

        self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads)

        return self
    
    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads):
        params = self.coefs_ + self.intercepts_

        self._optimizer = Adam(params, self.learning_rate, 
                            self.beta_1, self.beta_2, self.epsilon)

        
        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        batch_size = np.clip(self.batch_size, 1, n_samples)

        p = self.get_P(X)

        for it in range(self.iter_max):
            accumulated_loss = 0.0

            x_IS, y_IS = self.get_IS_sample(X, y, p)
            for batch_slice in gen_batches(len(x_IS), batch_size):
                X_batch = x_IS[batch_slice]
                y_batch = y_IS[batch_slice]

            # for batch_slice in gen_batches(n_samples, batch_size):
            #     X_batch = X[batch_slice]
            #     y_batch = y[batch_slice]

                activations[0] = X_batch
                batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch, y_batch, activations, deltas,
                        coef_grads, intercept_grads)
                accumulated_loss += batch_loss * (batch_slice.stop -
                                                  batch_slice.start)

                grads = coef_grads + intercept_grads
                self._optimizer.update_params(grads)
            
            self.n_iter_ += 1
            self.loss_ = accumulated_loss / X.shape[0]

            self.t_ += n_samples
            self.loss_p.append(self.loss_)
            if self.verbose:
                print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))
            
            self._update_no_improvement_count()

            if self._no_improvement_count > self.n_iter_no_change:
                msg = ("Training loss did not improve more than tol=%f"
                        " for %d consecutive epochs." % 
                        (self.tol, self.n_iter_no_change))

                is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                if is_stopping:
                    break
                else:
                    self._no_improvement_count = 0
    
    def predict(self, X):

        y_pred = self._forward_pass_fast(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred


np.random.seed(10)

#the first data set
data = pd.read_table("data1.txt", sep = '\s+' ,header=None)

#second
# data = pd.read_csv("energydata_complete.csv")

#third
# data = pd.read_csv("train.csv")

#forth
# data = pd.read_csv("blogData_train.csv")

data = np.array(data)

mm = MinMaxScaler()
data_s = mm.fit_transform(data)

train_set, test_set = train_test_split(data_s, test_size=0.2, random_state=0)
width = np.size(data[0])-1
x_train_1 = train_set[:,0:width]
length = len(x_train_1)
y_train_1 = train_set[:,width]  
fit1 = IS_Adam(hidden_size=(1000,200), alpha=0.001, batch_size=200, 
                learning_rate=0.001, iter_max=length, tol=1e-6, verbose=True, 
                beta_1=0.9,beta_2=0.999, epsilon=1e-8, n_iter_no_change=50)
print("fitting model right now")
fit1.fit(x_train_1, y_train_1)


x_test = test_set[:, 0:width]
y_test = mm.inverse_transform(test_set)[:, width]
pred1_test = fit1.predict(x_test)
pred1_test = pred1_test[:,np.newaxis]
ppp = test_set[:,-1][:,np.newaxis]
aaa = np.concatenate((x_test, pred1_test), axis=1)
y_pre1 = mm.inverse_transform(aaa)[:, width]
mse_1 = mean_squared_error(y_pre1, y_test)



#SGD
fit2 = MLPRegressor(
    hidden_layer_sizes=(1000,200), activation='relu',
    solver='sgd', alpha=0.001, max_iter=length, tol = 1e-6,
    verbose=True, learning_rate_init=0.0001, n_iter_no_change = 50,
    batch_size= 200)
fit2.fit(x_train_1, y_train_1)

pred2_test = fit2.predict(x_test)
pred2_test = pred2_test[:,np.newaxis]
ppp = test_set[:,-1][:,np.newaxis]
aaa = np.concatenate((x_test, pred2_test), axis=1)
y_pre2 = mm.inverse_transform(aaa)[:, width]
mse_2 = mean_squared_error(y_pre2, y_test)

fit3 = MLPRegressor(
    hidden_layer_sizes=(1000,200), activation='relu',
    solver='adam', alpha=0.001, max_iter=length, tol = 1e-6,
    verbose=True, learning_rate_init=0.0001, n_iter_no_change = 50,
    batch_size= 200)
fit3.fit(x_train_1, y_train_1)

pred3_test = fit3.predict(x_test)
pred3_test = pred3_test[:,np.newaxis]
ppp = test_set[:,-1][:,np.newaxis]
aaa = np.concatenate((x_test, pred3_test), axis=1)
y_pre3 = mm.inverse_transform(aaa)[:, width]
mse_3 = mean_squared_error(y_pre3, y_test)

print("Test ERROR = ", mse_1)
print("Test ERROR (SGD) = ", mse_2)
print("Test ERROR (Adam) = ", mse_3)

plt.figure(1)
plt.plot(y_pre1[:50], color='green', label='IS-Adam')
plt.plot(y_pre2[:50], color='blue', label='SGD')
plt.plot(y_pre3[:50], color='red', label='Adam')
plt.plot(y_test[:50] , color='black', label='testing targets')
plt.legend()

plt.figure(2)
plt.plot(fit1.loss_p, color='green', label='IS-Adam')
plt.plot(fit2.loss_curve_, color='blue', label='SGD')
plt.plot(fit3.loss_curve_, color='red', label='Adam')
plt.legend()

plt.show()