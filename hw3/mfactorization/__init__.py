from .mf import matrix_factorization as mf
from .mf import matrix_factorization


mf_default_config = {
    'lr': (2e-2, 2e-3),

    'n_epochs': 400,
    'single_epochs': 300,

    'sample_s': 0.05, 
    'batch_sample_s': 0.05,

    'ff': 0.005,

    'save_logs': False,
}
