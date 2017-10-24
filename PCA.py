class RPCA:
    def __init__(self, xp, lmd, rho, max_iter, stopcri):
        self.xp = xp
        self.lmd = lmd
        self.rho = rho
        self.max_iter = max_iter
        self.stopcri = stopcri

    def __call__(self, M, echo_iter=False):
        # initialize
        L = M
        S = self.xp.zeros_like(M)

        Y1 = self.xp.zeros_like(M)
        Y2 = self.xp.zeros_like(M)
        Y3 = self.xp.zeros_like(M)

        D1 = self.xp.zeros_like(M)
        D2 = self.xp.zeros_like(M)
        D3 = self.xp.zeros_like(M)

        # optimize
        for i in range(self.max_iter):
            # update L and S
            Lpre = L
            Spre = S
            L = (2 * (Y1 - D1) - (Y2 - D2) + (Y3 - D3)) / 3
            S = -2 * L + (Y1 - D1) + (Y3 - D3)

            # update Y1
            for c in range(M.shape[2]):
                U, k, V = self.xp.linalg.svd(L[:, :, c] + D1[:, :, c],
                                             full_matrices=False)
                K = self.xp.diag(self.xp.maximum(0, k - 1 / self.rho))
                Y1[:, :, c] = self.xp.dot(U, self.xp.dot(K, V))

            # update Y2
            Y2 = S + D2
            Y2 = (self.xp.sign(Y2) *
                  self.xp.maximum(self.xp.abs(Y2) - self.lmd / self.rho, 0))

            # update Y3
            Y3 = M

            # update dual variables
            D1 = D1 + L - Y1
            D2 = D2 + S - Y2
            D3 = D3 + L + S - Y3

            errors = self.xp.vstack((L - Lpre, S - Spre)).flatten()
            error = self.xp.sum(self.xp.linalg.norm(errors, 2))
            if echo_iter and (i % echo_iter) == echo_iter - 1:
                print('{} iter, error: {}'.format(i + 1, error))
            if error < self.stopcri:
                break
        if echo_iter:
            print('{} iters, error: {}'.format(i + 1, error))
        return L, S
