import pandas as pd
import numpy as np

default_config = {
    'n_latent': 64,
    'r': 0.001, 
}

def linear_comb(a, b, alpha):
    return a + (b - a) * alpha

def py_linear_scheduler(n_epochs, init, end):
    step = 1/n_epochs
    x = 0

    for _ in range(n_epochs):
        yield init + (end - init) * x
        x += step

def linear_scheduler(n_epochs, init, end):
    X = np.linspace(init, end, num=n_epochs, endpoint=True)
    
    for i in range(n_epochs):
        yield X[i]

def acc_scheduler(n_epochs, init, end):
    X = np.linspace(-2, -1, n_epochs, True)
    B = (2*init - end) / (end - init)
    A = end * (B + 1)
    return A / (B - X)

def learning_rate_scheduler(
    n_epochs,
    INIT_LR = 0.05,
    END_LR = 0.0001,
    warmup = 1.0/5,
    x = 0.3
):
    w_init = INIT_LR + (1 - x) * (END_LR - INIT_LR)
    n_warmup = int(n_epochs * warmup)

    # warmup first
    for lr in acc_scheduler(n_warmup, w_init, INIT_LR):
        yield lr

    n_epochs -= n_warmup

    # linear = linear_scheduler(n_epochs, INIT_LR, END_LR)
    x = np.exp(np.linspace(-1, 0, n_epochs, True))
    x = INIT_LR + (1-x) * END_LR

    for lr in x:
        yield lr

    return None


    n_warmup = warmup * n_epochs
    s = warmup

    for epoch in range(n_epochs):
        
        x = (epoch - n_epochs)/n_epochs
        
        if epoch >= n_warmup:
            x = x - s
            # y = ((1 + x))
            y = (n_epochs*(s-1) + 1/(1 - s))*x + n_epochs*(1-s)
        else:
            x += 1
            y = (-2./s)*(x**2) + 2*s
        
        lr = INIT_LR - (INIT_LR-END_LR) * y

        yield lr
