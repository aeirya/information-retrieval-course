from lrschedule import learning_rate_scheduler
from tqdm import tqdm
import numpy as np


rng = np.random.default_rng(seed=1234)

def init_pq(n_users, n_items, n_latent):
    scale = 1./np.sqrt(n_latent)/4
    loc = 0

    Q0 = rng.normal(loc=loc, scale=scale, size=(n_users, n_latent))
    P0 = rng.normal(loc=loc, scale=scale, size=(n_items, n_latent))

    return Q0, P0


def clip_negative(m):
    m[m < 0] = 0


def update_pq(r, q, p, reg, lr):
    e = (r - q @ p.T)
    qq = q + lr * (2 * e @ p - reg*q)
    pp = p + lr * (2 * e.T @ q - reg*p)

    return qq, pp


def error(r, q, p, reg):
    n = r.nonzero()
    return (r[n] - (q@p.T)[n]).sum() + reg*(q*q)[n[0]].sum() + reg*(p*p)[n[1]].sum()


def matrix_factorization(
        r,
        lr = (1e-3, 1e-5),
        n_epochs = 10000,
        
        reg = 0.001,
        n_latent = 64,
        log_step = 20,
        print_step = 100,
        eth = 1e-5,

        qp = None,
):
    err_check = lambda e : e < eth
    errc_step = min(log_step, print_step)
    loss = error

    n_users, n_items = r.shape
    q, p = qp if qp else init_pq(n_users, n_items, n_latent)
    
    init_lr, end_lr = lr if isinstance(lr, tuple) else (lr, lr)
    print(init_lr, end_lr)

    lr_scheduler = learning_rate_scheduler(n_epochs, init_lr, end_lr, reach_endpoint=True)
    iterator = zip(range(n_epochs), lr_scheduler)

    logs = []

    for epoch, lr in tqdm(iterator):
        q, p = update_pq(r, q, p, reg, lr)

        if epoch % errc_step == 0:
            err = loss(r, q, p, reg)
            
            if err_check(err):
                print(f"error less than {eth:.2e}")
                break

            if epoch % log_step == 0:
                logs.append((err, q, p))
        
            if epoch % print_step == 0:
                print(f'err: {err:.2e}, lr: {str(lr)[:9]}')

    err, q, p = min(logs)

    return q, p, err, logs


