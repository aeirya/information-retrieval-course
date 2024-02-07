import numpy as np

def acc_scheduler(n_epochs, init, end):
    if init == end:
        return []

    X = np.linspace(-2, -1, n_epochs, True)
    B = (2*init - end) / (end - init)
    A = end * (B + 1)
    return A / (B - X)

def learning_rate_scheduler(
    n_epochs,
    init_lr = 0.05,
    end_lr = 0.0001,
    warmup = 0.01,
    x = 0.1,
    reach_endpoint = False
):
    w_init = init_lr + (1 - x) * (end_lr - init_lr)
    n_warmup = int(n_epochs * warmup)

    # warmup first
    for lr in acc_scheduler(n_warmup, w_init, init_lr):
        yield lr

    n_epochs -= n_warmup

    # exponential decrease 
    if not reach_endpoint:
        n_exp = n_epochs*3//4
    else:
        n_exp = n_epochs

    x = np.exp(np.linspace(0.0, -3.0, n_exp, True))
    x = init_lr + (1.0 - x) * (end_lr-init_lr)

    for lr in x:
        yield lr

    n_epochs -= n_exp
    
    if reach_endpoint:
        # linear tail
        for lr in np.linspace(x[-1], end_lr, n_epochs, True):
            yield lr

    return None


'''
UNUSED CODE
'''
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
