from .mf import matrix_factorization as mf

mf_default_config = {
    'lr': (2e-2, 2e-3),

    'n_epochs': 700,
    'single_epochs': 100,

    'sample_s': 0.01, 
    'batch_sample_s': 0.05,
}
