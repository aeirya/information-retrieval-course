from .mf import matrix_factorization as mf
from .mf import matrix_factorization


mf_default_config = {
    'lr': (2e-2, 2e-3),

    'n_epochs': 800,
    'single_epochs': 100,

    'sample_s': 0.07, 
    'batch_sample_s': 0.2,
}
