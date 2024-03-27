import cvxpy as cp
import numpy as np

class QuadraticSolver:

    def __init__(self, N: int, P: np.ndarray, q: np.ndarray):
        self.P = P
        self.q = q
        self.N = N

        self.x = cp.Variable(N)
        self.prob = cp.Problem(cp.Minimize(cp.quad_form(self.x, self.P) - self.q.T @ self.x))

        self.prob.solve()


    def optimal_value(self):
        return self.prob.value

    def optimal_point(self):
        return self.x.value

