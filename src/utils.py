import numpy as np


def softmax(x, axis=1):
    """
    Implements a stabilized softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    NOTE:
    Do not import or use these packages: sklearn, scipy, sys, importlib.

    Note that np.atleast_2d will take an array of shape [K, ]
        and make it an array of shape [1, K]. Thus if you pass
        an array of shape [K, ] to softmax, you should leave
        axis=1 as the default.
    """
    # print(f"\n x = {x}")

    x = np.atleast_2d(x)

    # print(f"\n x new = {x}")

    z = x - np.max(x)
    # print(f"\n np.max(x) = {np.max(x)}")
    # print(f"\n z new = {z}")

    softmax = np.exp(z)/np.sum(np.exp(z), axis=axis, keepdims=True)
    # print(f"\n np.sum(z) = {np.sum(z)}")

    return softmax

    # raise NotImplementedError


def stable_log_sum(X):
    """
    Implement a stabilized log sum operation.

    When all elements of X are greater than -745, this will be equivalent to:
        >>> np.sum(np.log(np.sum(np.exp(X), axis=1)))

    However, for large negative values of X, your computer will represent
        `np.exp(X)` as 0.0; this is called underflow. Your implementation
        should avoid underflow in one of two ways:

    1. You can assume that X is an array of shape (K, 2) and approximate the
          sum by ignoring the smallest of the two values in each of the K rows.
          That is, while `log(a) + log(b) != log(a + b)`, if a is much bigger
          than b, `log(a)` is a good approximation for `log(a + b)`.
          Thus, `max([log(X[i, 0]), log(X[i, 1])])` provides a decent approximation
          for `log(exp(X[i, 0]) + exp(X[i, 1]))`. You will still need to sum
          over all rows.

    2. To exactly compute the sum without relying on the above approximation,
          you can use some clever math that relies on the properties of
          logarithms to avoid having to ever compute np.exp(X). See the
          following link:
          https://stackoverflow.com/questions/22009862/how-to-calculate-logsum-of-terms-from-its-component-log-terms/22385004

    NOTE:
    Do not import or use these packages: sklearn, scipy, sys, importlib.

    Args:
        X: an array of shape (K, 2) for some K
    Returns:
        sum(log(sum(exp(X), axis=1))), avoiding underflow
    """
    # print(f"\n X = {X}")
    # You can assume that this array is of shape (K, 2)
    assert X.shape[1] == 2 and len(X.shape) == 2

    sum = 0

    # print(f"\n 0 = {np.max([np.log(X[0, 0]), np.log(X[0, 1])])}")
    # print(f"\n 1 = {np.max([np.log(X[1, 0]), np.log(X[1, 1])])}")
    # print(f"\n 2 = {np.max([np.log(X[2, 0]), np.log(X[2, 1])])}")
    # print(f"\n 3 = {np.max([np.log(X[3, 0]), np.log(X[3, 1])])}")

    # print(f"\n Return = {np.sum(np.log(np.sum(np.exp(X), axis=1)))}")

    print(f"\n np.min(X) = {np.min(X)}")

    if np.min(X) > -745:
        sum = np.sum(np.log(np.sum(np.exp(X), axis=1)))
    else:
        for i in range(X.shape[0]):
            if np.max([X[i, 0], X[i, 1]]) > -745:
                sum += np.sum(np.log(np.sum(np.exp(X[i]))))
            else:
                sum += np.max([X[i, 0], X[i, 1]])

    # print(f"\n sum = {sum}")

    return sum
    # raise NotImplementedError
