'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        
class MLPPolicy(Policy):
    """
    MLP policy class that computes action.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.hidden1=64
        self.hidden2=64
        self.weights = np.zeros((self.ob_dim+1)*self.hidden1 +
                                (self.hidden1+1)*self.hidden2 +
                                (self.hidden2+1)*self.ac_dim, dtype=np.float64)

        self.end_w1 = self.ob_dim*self.hidden1
        self.end_w1_with_bias = (self.ob_dim+1)*self.hidden1

        self.end_w2 = self.end_w1_with_bias + self.hidden1*self.hidden2
        self.end_w2_with_bias = self.end_w1_with_bias + (self.hidden1+1)*self.hidden2

        self.end_w3 = self.end_w2_with_bias + self.hidden2*self.ac_dim
        self.end_w3_with_bias = self.end_w2_with_bias + (self.hidden2+1)*self.ac_dim

        assert self.end_w3_with_bias == self.weights.shape[0]

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)

        w1 = self.weights[0:self.end_w1].reshape(self.ob_dim, self.hidden1)
        b1 = self.weights[self.end_w1:self.end_w1_with_bias]

        w2 = self.weights[self.end_w1_with_bias:self.end_w2].reshape(self.hidden1, self.hidden2)
        b2 = self.weights[self.end_w2:self.end_w2_with_bias]

        w3 = self.weights[self.end_w2_with_bias:self.end_w3].reshape(self.hidden2, self.ac_dim)
        b3 = self.weights[self.end_w3:self.end_w3_with_bias]

        layer1 = np.dot(ob, w1) + b1
        tanh_layer1 = np.tanh(layer1)
        layer2 = np.dot(tanh_layer1, w2) + b2
        tanh_layer2 = np.tanh(layer2)
        layer3 = np.dot(tanh_layer2, w3) + b3

        return np.tanh(layer3)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

