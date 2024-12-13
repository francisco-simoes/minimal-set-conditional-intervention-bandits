class RandomVariable:
    def __init__(self, distribution, *params, **kparams):
        """
        Initialize the random variable with a specified scipy distribution.

        Parameters:
        distribution (scipy.stats.rv_continuous or rv_discrete): The scipy distribution.
        *params: Parameters specific to the distribution (e.g., p for Bernoulli, loc
            and scale for normal).
        """
        self.distribution = distribution
        self.rv = distribution(*params, **kparams)
        self.domain = (
            self._build_discrete_domain()
            if hasattr(self.rv, "pmf")
            else self.rv.support()
        )

    def _build_discrete_domain(self):
        a, b = self.rv.support()
        domain = tuple(range(a, b + 1))
        return domain

    def sample(self, size=1):
        """
        Generate samples from the distribution along with their probabilities
        (or densities for continuous distributions).

        Parameters:
        size (int): Number of samples to generate. Default is 1.

        Returns:
        list of tuples: Each tuple contains (sampled_value, probability_of_sampled_value).
        """
        # Generate the samples
        samples = self.rv.rvs(size=size)

        prob = self.rv.pmf(samples) if hasattr(self.rv, "pmf") else self.rv.pdf(samples)

        return samples, prob


# class BernoulliRV:
#     def __init__(self, p):
#         """
#         Initialize a Bernoulli random variable.

#         Parameters:
#         p (float): Probability of success (1). Must be between 0 and 1.
#         """
#         if not (0 <= p <= 1):
#             raise ValueError("Probability p must be between 0 and 1.")
#         self.p = p

#     def sample(self, size=1):
#         """
#         Generate samples from the Bernoulli distribution.

#         Parameters:
#         size (int): Number of samples to generate. Default is 1.

#         Returns:
#         numpy array: An array of 0s and 1s sampled from the Bernoulli distribution.
#         """
#         return bernoulli.rvs(self.p, size=size)
