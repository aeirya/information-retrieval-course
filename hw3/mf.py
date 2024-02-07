from lrschedule import learning_rate_scheduler
from tqdm import tqdm
import numpy as np


rng = np.random.default_rng(seed=1234)

def init_pq(n_users, n_items, n_latent):  
    scale = 1./np.sqrt(n_latent)
    loc = 0

    Q0 = rng.normal(loc=loc, scale=scale, size=(n_users, n_latent))
    P0 = rng.normal(loc=loc, scale=scale, size=(n_items, n_latent))

    # Q0[np.abs(Q0) > scale] = 0
    # P0[np.abs(P0) > scale] = 0

    # s = np.sqrt(n_latent)
    # s = 1
    # Q0 = (rng.random(size=(n_users, n_latent))) / s
    # P0 = (rng.normal(size=(n_items, n_latent))) / s

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
    # batch_sizes = [2000, 1000, 500, 400, 300, 200, 100, 70, 50, 40, 30, 25, 20, 15, 12, 10, 5, 4, 4, 3] + 4*[3] + 4*[2] + 4*[1]
    # batch_sizes = [100] + [50]*3 + [30]*2 + [15]*8 + [5] * 4 + sorted([4, 3, 3,2, 2, 2, 2] * 2, reverse=True) + [1] * 8
    N = 8
    batch_sizes = 2 ** np.arange(N)
    batch_sizes = [[batch_sizes[i]]*((N-i+1)//2) for i in range(N-1)]
    from functools import reduce
    batch_sizes = reduce(lambda a,b: a+b, batch_sizes, [])
    batch_sizes = batch_sizes[::-1]
    return batch_sizes + [1]

def matrix_factorization(
        r,
        lr = (1e-2, 2e-3),
        n_epochs = 10000,
        
        reg = 0.001,
        n_latent = 64,
        log_step = 10,
        print_step = 100,
        eth = 1e-3,

        qp = None,
        sample_s = 0.01,
        batch_sample_s = 0.05,
        ff = 0.1,
):
    err_check = lambda e : np.abs(e) < eth
    errc_step = min(log_step, print_step)
    assert log_step % errc_step == 0 and print_step % errc_step == 0
    loss = error

    n_users, n_items = r.shape
    q, p = qp if qp else init_pq(n_users, n_items, n_latent)
    
    init_lr, end_lr = lr if isinstance(lr, tuple) else (lr, lr)
    # print(init_lr, end_lr)

    lr_scheduler = learning_rate_scheduler(n_epochs, init_lr, end_lr, reach_endpoint=True)
    iterator = zip(range(n_epochs), lr_scheduler)

    logs = []

    # users = rng.choice(n_users, 30, False)
    # items = rng.choice(n_items, 10, False)

    batch_sizes = get_batch_sizes()

    n_ff = max(20, min(150, int(ff*n_epochs))) # n fastforward
    batch_bucket_size = ((n_epochs-n_ff) // len(batch_sizes))+1
    batch = None

    for epoch, lr in tqdm(iterator):
        x = epoch-n_ff
        if x % batch_bucket_size == 0 and x >= 0:
            batch = batch_sizes[x // batch_bucket_size]
        
        # heatmap(r[0:30, 0:20])
        # shuffle_update(r, q, p, reg, lr)
        # heatmap(r[0:30, 0:20])
        # break

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
                logs.append((err, q, p))
        
            if epoch % print_step == 0:
                print(f'err: {err:.2e}, lr: {str(lr)[:9]}, last batch: {batch}')

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