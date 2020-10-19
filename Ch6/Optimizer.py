#!/usr/bin/env python
# coding: utf-8

# In[1]:


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# In[2]:


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - lr * grads[key]
            params[key] += self.v[key]


# In[3]:


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h += grads[key]*grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h) + 1e-7)


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def sigomid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100 # 각 은닉층의 노드(뉴런) 수
hideen_layer_size = 5 # 은닉층이 5개
activations = {} # 이곳에 활성화 결과(활성화 값)를 저장

for i in range(hideen_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num)  * 1
    a = np.dot(x, w)
    z = sigomid(a)
    activations[i] = z


# In[5]:


# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()


# In[6]:


x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100 # 각 은닉층의 노드(뉴런) 수
hideen_layer_size = 5 # 은닉층이 5개
activations = {} # 이곳에 활성화 결과(활성화 값)를 저장

for i in range(hideen_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num)  * 0.01
    a = np.dot(x, w)
    z = sigomid(a)
    activations[i] = z


# In[7]:


# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()


# In[8]:


x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100 # 각 은닉층의 노드(뉴런) 수
hideen_layer_size = 5 # 은닉층이 5개
activations = {} # 이곳에 활성화 결과(활성화 값)를 저장

for i in range(hideen_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    z = sigomid(a)
    activations[i] = z


# In[9]:


# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

