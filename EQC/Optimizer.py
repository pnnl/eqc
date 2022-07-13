import numpy as np


class Adam():
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = learning_rate

    def update(self, t, w, dw):
        # ========================================================================================
        # Update rule for ADAM Optimizer
        # ========================================================================================
        t = t+1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        return w


class SGD():
    def __init__(self,learning_rate=1):
        self.learning_rate = learning_rate

    def update(self,epoch,w,dw,z):
        # ========================================================================================
        # Update rule for SGD Optimizer
        # ========================================================================================
        print(f"Updating with gradient {dw} with Z: {z}")
        new_value =  w + z*self.learning_rate*dw
        return new_value[0]
