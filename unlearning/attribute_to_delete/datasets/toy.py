import numpy as np


def toy_gaussians(N, d, mu1=1.0, std1=2.0, mu2=-1.0, std2=2.0) -> tuple:
    X = np.concatenate(
        [
            np.random.randn(N // 2, d) * std1 + mu1,
            np.random.randn(N // 2, d) * std2 + mu2,
        ]
    )
    Y = np.concatenate([-np.ones(N // 2), np.ones(N // 2)]).astype(np.int32)

    X_test = np.concatenate(
        [
            np.random.randn(N // 2, d) * std1 + mu1,
            np.random.randn(N // 2, d) * std2 + mu2,
        ]
    )
    Y_test = np.concatenate([-np.ones(N // 2), np.ones(N // 2)]).astype(np.int32)
    return X, Y, X_test, Y_test
