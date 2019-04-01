"""
Implementation of BFGS.

Greatly inspired by:
    https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
"""

import sys
import time
import numpy as np
import numdifftools as nd
from .helpers import ls_wolfe12
from scipy.optimize.optimize import OptimizeResult

class BFGS:

    def __init__(self, f, x0, grad=None, hess=None, **kwargs):
        """
        Minimize using BFGS.

        :param f: objective function
        :param x0: initial point
        :param grad: Gradient of f. If None, we use numdifftools
        :param hess: Hessian of f. If None, we use numdifftools
        :param kwargs: Contains some other parameters
        """

        # Main parameters provided by the user
        self.func = f
        self.x0 = x0

        if grad is None:
            self.grad = nd.Gradient(f)
        else:
            self.grad = grad

        if hess is None:
            self.hess = nd.Hessian(f)
        else:
            self.hess = hess

        # Some variables used for the algorithm
        self.status = '?'
        self.n = len(x0)

        # Some other parameters in kwargs
        self.nbr_epochs = kwargs.get('nbr_epochs', 20)
        self.thresh = kwargs.get('thresh', 1.0e-6)
        self.verbose = kwargs.get('verbose', False)
        self.seed = kwargs.get('seed', -1)
        self.biogeme = kwargs.get('biogeme', None)
        self.bounds = kwargs.get('bounds', None)

        # Formats to display info about subproblem
        self.hd_fmt = '     %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = '     %-5d  %9.2e  %8.2e\n'

        # Return values
        self.xs = []
        self.epochs = []
        self.fs = []

        self.f = None
        self.x = None
        self.ep = None
        self.it = None
        self.optimized = None
        self.opti_time = None

    def solve(self, maximize=False):
        """
        Minimize the objective function f starting from x0.

        :return: x: the optimized parameters
        """
        start_time = time.clock()

        if self.verbose:
            if maximize:
                solving = 'Maximizing'
            else:
                solving = 'Minimizing'

            self._write("{} the problem using Newton Method\n".format(solving))

        xk = np.asarray(self.x0).flatten()

        self.ep = 0
        self.it = 0
        I = np.eye(len(self.x0))
        Bk = I

        mult = 1
        if maximize:
            mult = -1

        if self.biogeme is None:
            fprime = lambda x: mult*self.grad(x)
        else:
            fprime = lambda x: mult*self.biogeme.calculateLikelihoodAndDerivatives(x, hessian=False)[1]

        f = lambda x: mult*self.func(x)

        self.xs = []
        self.epochs = []
        self.fs = []

        while self.ep < self.nbr_epochs:

            fk = f(xk)
            gk = fprime(xk)

            self.xs.append(xk)
            self.fs.append(fk)
            self.epochs.append(self.ep)

            if self.verbose:
                self._write("Epoch {}:\n".format(self.ep))
                self._write("  xk = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                self._write("  f(xk) = {:.3f}\n".format(fk))

            step = -np.dot(Bk, gk)

            if self.it > 0:
                old_old_fval = self.fs[-2]
            else:
                old_old_fval = self.fs[-1] + np.linalg.norm(gk) / 2

            alpha = ls_wolfe12(f, fprime, xk, step, gk, self.fs[-1], old_old_fval)

            xkp1 = xk + alpha * step
            sk = xkp1 - xk
            xk = xkp1

            gkp1 = fprime(xkp1)
            yk = gkp1 - gk
            gk = gkp1

            gnorm = np.linalg.norm(gk)
            snorm = np.linalg.norm(step)

            if self.verbose:
                self._write("  ||gk|| = {:.3E}\n".format(gnorm))
                self._write("  ||step|| = {:.3E}\n".format(np.linalg.norm(step)))
                self._write("  alpha = {:.3E}\n".format(alpha))

            if (gnorm <= self.thresh) or (snorm <= self.thresh):
                self.status = 'Optimum reached!'
                if self.verbose:
                    self._write("Algorithm Optimized!\n")
                    self._write("  x* = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
                    self._write("  f(x*) = {:.3f}\n".format(fk))

                self.optimized = True
                break

            try:  # this was handled in numeric, let it remains for more safety
                rhok = 1.0 / (np.dot(yk, sk))
            except ZeroDivisionError:
                rhok = 1000.0
            if np.isinf(rhok):  # this is patch for numpy
                rhok = 1000.0
            A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            Bk = np.dot(A1, np.dot(Bk, A2)) + (rhok * sk[:, np.newaxis] *
                                                     sk[np.newaxis, :])

            if self.verbose:
                self._write('\n')

            if self.bounds is not None:
                tmp = []
                for i, x in enumerate(xk):
                    if self.bounds[i][1] is not None and x > self.bounds[i][1]:
                        tmp.append(self.bounds[i][1])
                    elif self.bounds[i][0] is not None and x < self.bounds[i][0]:
                        tmp.append(self.bounds[i][0])
                    else:
                        tmp.append(x)
                xk = tmp

            self.it += 1
            self.ep += 1

        if not self.optimized:
            self.status = 'Optimum not reached'

        if self.verbose and not self.optimized:
            self._write("Algorithm not fully optimized!\n")
            self._write("  x_n = [{}]\n".format(", ".join(format(x, ".3f") for x in xk)))
            self._write("  f(x_n) = {:.3f}\n".format(fk))

        self.f = mult*f(xk)
        self.x = xk
        gk = fprime(xk)

        self.opti_time = time.clock() - start_time

        dct = {'x': self.x,
               'success': self.optimized,
               'status': self.status,
               'fun': self.f,
               'jac': gk,
               'hess': Bk,
               'nit': self.it,
               'nep': self.ep,
               'opti_time': self.opti_time}

        return OptimizeResult(dct)

    def _write(self, msg):
        sys.stderr.write(msg)