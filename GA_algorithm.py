import numpy as np



class GA():
    def __init__(self, n_params,
        sigma_init=0.01,
        sigma_decay=0.999,
        sigma_limit = 0.001,
        n_elites = 16,
        popsize=256,

        ):




        self.n_params = n_params
        self.sigma = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.n_elites = n_elites
        self.popsize = popsize

        self.elite_params = np.zeros((n_elites, n_params))

        self.best_score = -np.inf

    def ask(self):

        ##populate with elites
        solutions = np.repeat(self.elite_params, self.popsize//len(self.elite_params), axis=0)

        ##crossover
        solutions_b = np.copy(solutions)
        np.random.shuffle(solutions_b)
        cross = np.arange(self.n_params)
        np.random.shuffle(cross)
        cross = cross[:self.n_params//2]

        solutions[:,cross] = solutions_b[:,cross]

        ##random mutations
        epsilon = np.random.normal( 0.,self.sigma, (self.popsize, self.n_params))
        self.solutions = solutions + epsilon

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

        return self.solutions


    def tell(self,rewards):
        idx = np.argsort(rewards)[::-1][0:self.n_elites]

        elite_rewards = rewards[idx]
        self.elite_params = self.solutions[idx]
        if elite_rewards[0] >= self.best_score:
            self.best_score = elite_rewards[0]
            self.best_solition = self.elite_params[0]



