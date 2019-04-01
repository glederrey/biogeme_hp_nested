import numpy as np
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2, line_search_armijo


def ls_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval)
    alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('A')
        # line search failed: try different one.
        ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                 old_fval, old_old_fval)
        alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('B')
        ret = line_search_armijo(f, xk, pk, gfk, old_fval)
        alpha = ret[0]

    if alpha is None or alpha < 1e-12:
        #print('C')
        alpha = backtracking_line_search(f, gfk, xk, pk)

    return alpha


def cg(xk, A, b):
    """
    Conjugate gradient taken from the function _minimize_newtoncg in the package scipy.optimize.optimize.

     See https://github.com/scipy/scipy/blob/v1.1.0/scipy/optimize/optimize.py line 1471
     """

    float64eps = np.finfo(np.float64).eps

    cg_maxiter = 20*len(xk)

    maggrad = np.add.reduce(np.abs(b))
    eta = np.min([0.5, np.sqrt(maggrad)])
    termcond = eta * maggrad

    xsupi = np.zeros(len(xk))

    ri = -b
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)

    for k2 in range(cg_maxiter):
        if np.add.reduce(np.abs(ri)) <= termcond:
            break
        Ap = np.dot(A, psupi)
        # check curvature
        Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 3 * float64eps:
            break
        elif curv < 0:
            if (i > 0):
                break
            else:
                # fall back to steepest descent direction
                xsupi = dri0 / (-curv) * b
                break
        alphai = dri0 / curv
        xsupi = xsupi + alphai * psupi
        ri = ri + alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        i = i + 1
        dri0 = dri1  # update np.dot(ri,ri) for next time.

    return xsupi


def backtracking_line_search(func, g, x, step):
    ll = func(x)

    m = np.dot(np.transpose(step), g)

    c = 0.9

    t = -c * m
    tau = 0.5

    alpha = 10

    while True:
        tmp_x = x + alpha * step
        if np.abs(ll - func(tmp_x)) <= alpha * t or alpha < 1e-8:
            return alpha

        alpha = tau * alpha


def line_search(func, x, step):
    ll = func(x)

    alpha = 5

    while True:
        tmp_x = x + alpha * step
        if func(tmp_x) < ll or alpha < 1e-8:
            return alpha

        alpha = alpha/2


def conjgrad(A, b, x):
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def wma(values, curr_iter, window_size):

    curr_iter = int(curr_iter)
    if curr_iter < window_size:
        window_size = curr_iter

    window_size = int(window_size)

    tmp = [(i + 1) * v for i, v in enumerate(values[curr_iter - window_size:curr_iter])]

    wma = np.sum(tmp) / np.sum(range(1, window_size + 1))

    return wma


def ema(values, curr_iter, window_size, decay=1):

    if curr_iter < window_size:
        window_size = curr_iter

    weights = np.exp(np.linspace(-decay, 0., window_size))
    weights /= weights.sum()

    tmp = [v * w for v, w in zip(values[curr_iter - window_size:curr_iter], weights)]

    ema = np.sum(tmp)

    return ema


def stop_crit(xs, f, grad):

    vals = [np.abs(x*df) for x, df in zip(xs, grad)]

    return np.max(vals)/f


def back_to_bounds(xk, bounds):
    if bounds is not None:
        tmp = []
        for i, x in enumerate(xk):
            tmp.append(min(max(x, bounds[i][0]), bounds[i][1]))

        return tmp
    else:
        return xk