"""Contains functions for solving systems of equations."""

import numpy as np



def tdma(x, a, b, c, d) -> np.ndarray:
    """Solves for x given a tridiagonal matrix, following Thomas' Algorithm."""

    # forward substitution
    p = np.zeros_like(x, float)
    q = np.zeros_like(x, float)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, x.size):
        dn = b[i] + a[i]*p[i - 1]
        p[i] = - c[i] / dn
        q[i] = (d[i] - a[i]*q[i - 1]) / dn

    # back substitution for x
    x[-1] = q[-1]
    for i in range(x.size - 2, -1, -1):
        x[i] = p[i]*x[i + 1] + q[i]

    return x
