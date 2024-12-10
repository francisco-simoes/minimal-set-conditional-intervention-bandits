from scipy.stats import bernoulli


class RandomVariable:
    def __init__(self, distribution, *params):
        """
        Initialize the random variable with a specified scipy distribution.

        Parameters:
        distribution (scipy.stats.rv_continuous or rv_discrete): The scipy distribution.
        *params: Parameters specific to the distribution (e.g., p for Bernoulli, loc
            and scale for normal).
        """
        self.distribution = distribution
        self.params = params

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
        samples = self.distribution.rvs(*self.params, size=size)

        # Compute probabilities/densities for each sampled value
        # if isinstance(samples, np.ndarray):
        #     results = [
        #         (
        #             sample,
        #             self.distribution.pmf(sample, *self.params)
        #             if hasattr(self.distribution, "pmf")
        #             else self.distribution.pdf(sample, *self.params),
        #         )
        #         for sample in samples
        #     ]
        # else:
        prob = (
            self.distribution.pmf(samples, *self.params)
            if hasattr(self.distribution, "pmf")
            else self.distribution.pdf(samples, *self.params)
        )

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
