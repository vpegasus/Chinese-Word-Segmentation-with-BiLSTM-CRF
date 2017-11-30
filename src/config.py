
config = {
    'data_dir': '/hdd/data/Sighan-2004',
    'seed': 9788,

    'window_size': 50,
    'window_step': 25,

    'batch_size': 128,
    'test_batch_size': 16,

    'embedding_size': 300,

    'hidden_size': 200,
    'n_layers':3,

    'use_dropout': True,
    'dropout_rate': 0.6,
    'bidirectional': True,

    'linear': [100, 4],
    'init_range': 0.1,
}