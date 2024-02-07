from lrschedule import learning_rate_scheduler
from tqdm import tqdm
import numpy as np


rng = np.random.default_rng(seed=1234)

def init_pq(n_users, n_items, n_latent):  
    scale = 1./np.sqrt(n_latent) / 4
    loc = 0

    Q0 = rng.normal(loc=loc, scale=scale, size=(n_users, n_latent))
    P0 = rng.normal(loc=loc, scale=scale, size=(n_items, n_latent))

    # s = np.sqrt(n_latent)
    # Q0 = (rng.random(size=(n_users, n_latent)) - 0.5) / s
    # P0 = (rng.normal(size=(n_items, n_latent)) - 0.5) / s

    # Q0 = np.zeros((n_users, n_latent))
    # P0 = np.zeros((n_items, n_latent))

    return Q0, P0


def clip_negative(m):
    m[m < 0] = 0

def clip(m, a):
    a = np.abs(a)
    m[m > a] = a
    a = -a
    m[m < a] = a


def update_pq(r, q, p, reg, lr):
    clip(q, 2.5)
    clip(p, 2.5)

    e = (r - q @ p.T)
    qq = q + lr * (e @ p - reg*q)
    pp = p + lr * (e.T @ q - reg*p)

    q[:,:] = qq
    p[:,:] = pp

    return e


def update_pg_sampled(r, q, p, reg, lr, sample):
    i,j = sample
    nu = q.shape[0]
    ni = p.shape[0]

    e = r[i][:, j] - q[i] @ p[j].T
    qq = q[i] + lr * (e @ p[j] - reg*q[i])
    pp = p[j] + lr * (e.T @ q[i] - reg*p[j])

    clip(qq, 2)
    clip(pp, 2)
    
    q[i, :] = qq
    p[j, :] = pp

    return e, r[i][:, j]


def sampled_update(r, q, p, reg, lr):
    n = r.nonzero()
    n_nonzero = n[0].shape[0]
    samples = rng.choice(n_nonzero, (4, 400), False)
    E = []
    R = []
    for i in range(4):
        s = samples[i]
        e, rs = update_pg_sampled(r, q, p, reg, lr, (n[0][s], n[1][s]))
        E.append(e)
        R.append(rs)

    return np.mean(E, axis=0), np.mean(R, axis=0)


def error(r, q, p, reg):
    n = r.nonzero()
    e = (r[n] - (q@p.T)[n]).sum() + reg*(q*q)[n[0]].sum() + reg*(p*p)[n[1]].sum()
    return e


def heatmap(a):
    import matplotlib.pyplot as plt
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()


def shuffle(m, j=None, rows=True):
    n = m.shape[0 if rows else 1]
    i = np.arange(n)

    if j is None:
        j = rng.choice(n, n, False)

    if rows:
        m[j] = m[i]
    else:
        m[:, j] = m[:, i]
    
    return j

def revert(m, j, rows=True):
    n = m.shape[0 if rows else 1]
    i = np.arange(n)

    if rows:
        m[i] = m[j]
    else:
        m[:, i] = m[:, j]


def update_shuffled_rows(r, q, p, reg, lr, i0, i1):
    r = r[i0:i1]
    q = q[i0:i1]
    e = (r - q @ p.T)
    qq = q + lr * (e @ p - reg*q)
    pp = p + lr * (e.T @ q - reg*p)

    q[:,:] = qq
    p[:,:] = pp


def shuffle_update(r, q, p, reg, lr):
    clip(q, 2.5)
    clip(p, 2.5)

    j = shuffle(r)
    shuffle(q, j)

    u = r.shape[0]
    step = 500
    for i in range(0, 3000, step):
        update_shuffled_rows(r, q, p, reg, lr, i, i+step)

    revert(r, j)
    revert(q, j)


def single_update(r, q, p, reg, lr):
    for i, j in zip(*r.nonzero()):
        e = r[i,j] - q[i]@p[j].T
        qq = q[i] + lr * (e * p[j] - reg*q[i])
        pp = p[j] + lr * (e * q - reg*p[j])
        
        q[i,j] = qq
        p[i,j] = pp
    

def matrix_factorization(
        r,
        lr = (1e-3, 1e-5),
        n_epochs = 10000,
        
        reg = 0.001,
        n_latent = 64,
        log_step = 20,
        print_step = 100,
        eth = 1e-3,

        qp = None,
):
    err_check = lambda e : np.abs(e) < eth
    errc_step = min(log_step, print_step)
    loss = error

    n_users, n_items = r.shape
    q, p = qp if qp else init_pq(n_users, n_items, n_latent)
    
    init_lr, end_lr = lr if isinstance(lr, tuple) else (lr, lr)
    print(init_lr, end_lr)

    lr_scheduler = learning_rate_scheduler(n_epochs, init_lr, end_lr, reach_endpoint=True)
    iterator = zip(range(n_epochs), lr_scheduler)

    logs = []

    users = rng.choice(n_users, 30, False)
    items = rng.choice(n_items, 10, False)

    for epoch, lr in tqdm(iterator):
        shuffle_update(r, q, p, reg, lr)
        
        if epoch % errc_step == 0:
            err = loss(r, q, p, reg)
            
            if err_check(err):
                print(f"error {err:.2e} less than {eth:.2e}")
                break

            if epoch % log_step == 0:
                logs.append((err, q, p))
        
            if epoch % print_step == 0:
                print(f'err: {err:.2e}, lr: {str(lr)[:9]}')

    err, q, p = min(logs)

    return q, p, err, logs




           # heatmap(np.concatenate((errors[users[:, None], items], np.ones((users.shape[0], 4)), r[users[:, None], items]), axis=1))  
        # else:
        #     errors, rs = sampled_update(r, q, p, reg, lr)
        #     if epoch % (print_step*2) == 0:
        #         pass
        #         # heatmap(np.concatenate((errors, np.ones((rs.shape[0], 3)), rs)))
        #         # heatmap(errors)
        #         # heatmap(rs)