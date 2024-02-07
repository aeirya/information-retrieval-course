from lrschedule import learning_rate_scheduler
from tqdm import tqdm
import numpy as np


rng = np.random.default_rng(seed=1234)


def init_pq(n_users, n_items, n_latent):  
    scale = 1./np.sqrt(n_latent)
    loc = 0

    Q0 = rng.normal(loc=loc, scale=scale, size=(n_users, n_latent))
    P0 = rng.normal(loc=loc, scale=scale, size=(n_items, n_latent))

    return Q0, P0


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

    e = r[i][:, j] - q[i] @ p[j].T
    qq = q[i] + lr * (e @ p[j] - reg*q[i])
    pp = p[j] + lr * (e.T @ q[i] - reg*p[j])

    clip(qq, 2)
    clip(pp, 2)
    
    q[i, :] = qq
    p[j, :] = pp

    return e, r[i][:, j]


def sampled_update(r, q, p, reg, lr, rounds=4, batch=50, sub=0.2):
    n = r.nonzero()
    n_nonzero = n[0].shape[0]

    if rounds < 0:
        rounds = n_nonzero // batch
    if sub > 0:
        rounds = int(rounds * sub)

    samples = rng.choice(n_nonzero, (rounds, batch), False)
    E = []
    R = []
    for i in range(rounds):
        s = samples[i]
        e, rs = update_pg_sampled(r, q, p, reg, lr, (n[0][s], n[1][s]))
        E.append(e)
        R.append(rs)

    return np.mean(E, axis=0), np.mean(R, axis=0)


def error(r, q, p, reg=0.001):
    n = r.nonzero()
    e = (r[n] - (q@p.T)[n]).sum() + reg*(q*q)[n[0]].sum() + reg*(p*p)[n[1]].sum()
    return e


def error_pq(r, qp, q, p, reg=0.001):
    n = r.nonzero()
    e = (r[n] - qp[n]).sum() + reg*(q*q)[n[0]].sum() + reg*(p*p)[n[1]].sum()
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


def shuffle_update(r, q, p, reg, lr, clipth=2.5):
    clip(q, clipth)
    clip(p, clipth)

    j = shuffle(r)
    shuffle(q, j)

    k = shuffle(r, rows=False)
    shuffle(p, k)

    u = r.shape[0]
    step = u // 7
    for i in range(0, u - step * 2, step):
        update_shuffled_rows(r, q, p, reg, lr, i, i+step)

    revert(r, k, False)
    revert(p, k)
    revert(r, j)
    revert(q, j)


def sampled_single_update(r, q, p, reg, lr, sub=0.05):
    n = r.nonzero()[0].shape[0]
    non = r.nonzero()
    s = int(n * sub) if sub > 0 else n
    sample = rng.choice(n, s, False)

    for i, j in zip(non[0][sample], non[1][sample]):
        e = r[i,j] - q[i]@p[j].T
        qq = q[i] + lr * (e * p[j] - reg*q[i])
        pp = p[j] + lr * (e * q[i] - reg*p[j])
        
        q[i,:] = qq
        p[j,:] = pp
    

def get_batch_sizes():
    N = 8
    batch_sizes = 2 ** np.arange(N)
    batch_sizes = [[batch_sizes[i]]*((N-i+1)//2) for i in range(N-1)]
    from functools import reduce
    batch_sizes = reduce(lambda a,b: a+b, batch_sizes, [])
    batch_sizes = batch_sizes[::-1]
    return batch_sizes + [1]


def infer(q, p, th=0.2):
    U, I = q.shape[0], p.shape[0]
    M = np.zeros((U, I))
    S = q @ p.T
    S -= np.mean(S)
    S /= max(S.min(), S.max())
    M[S>th] = 1
    
    return M


def matrix_factorization(
        r,
        
        lr = (1e-2, 2e-3),
        n_epochs = 10000,
        
        reg = 0.001,
        n_latent = 64,

        log_step = -1,
        print_step = -1,
        eth = 1e-2,

        qp = None,
        sample_s = 0.01,
        batch_sample_s = 0.05,
        ff = 0.05,
):
    if log_step < 0:
        log_step = n_epochs // 50

    if print_step < 0:
        print_step = log_step * 5
    
    
    errc_step = min(log_step, print_step)
    assert log_step % errc_step == 0 and print_step % errc_step == 0

    print('log step', log_step)

    err_check = lambda e : np.abs(e) < eth
    loss = error

    n_users, n_items = r.shape
    q, p = qp if qp else init_pq(n_users, n_items, n_latent)
    
    init_lr, end_lr = lr if isinstance(lr, tuple) else (lr, lr)

    lr_scheduler = learning_rate_scheduler(n_epochs, init_lr, end_lr, reach_endpoint=True)
    iterator = zip(range(n_epochs), lr_scheduler)

    logs = []

    batch_sizes = get_batch_sizes()

    n_ff = max(20, min(150, int(ff*n_epochs))) # n fastforward
    batch_bucket_size = ((n_epochs-n_ff) // len(batch_sizes))+1
    batch = None

    for epoch, lr in tqdm(iterator):
        x = epoch-n_ff
        if x % batch_bucket_size == 0 and x >= 0:
            batch = batch_sizes[x // batch_bucket_size]
        
        if epoch < 0.85 * n_ff:
            update_pq(r, q, p, reg, lr)
        elif epoch < n_ff:
            shuffle_update(r, q, p, reg, lr)
        elif batch == 1:
            sampled_single_update(r, q, p, reg, lr, sub=sample_s)
        else:
            sampled_update(r, q, p, reg, lr, rounds=-1, batch=batch, sub=batch_sample_s)

        if epoch % errc_step == 0:
            err = loss(r, q, p, reg)
            
            if err_check(err):
                print(f"error {err:.2e} less than {eth:.2e}")
                break

            if epoch % log_step == 0:
                logs.append((err, q.copy(), p.copy()))
        
            if epoch % print_step == 0:
                print(f'err: {err:.2e}, lr: {str(lr)[:9]}, last batch: {batch}')

    err, q, p = min(logs)

    return q, p, err, logs
