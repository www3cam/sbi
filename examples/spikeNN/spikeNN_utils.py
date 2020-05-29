import torch
from typing import Optional, List
import numpy as np
from scipy.stats import norm
import scipy.linalg as sl


def calc_summ_stats_numpy(x, lags=[1, 10, 50, 100, 200, 500, 1000], norm=True):
    """
    Calculate and returns the following summary statistics for input trace: mean,
    variance, autocorrelation for input lags, crosscorrelation.
    Returns numpy matrix of size samples x number of summary statistics x neurons

    :param x: numpy matrix containing the rate trace of shape samples x timebins x
    neurons
    :param lags: list containing lags (in number of timebins) at which autocorrelation
    is to be calculated. Default is [1, 10, 50, 100, 200, 500, 1000]. 
    :param norm: boolean value. Set to True if autocorrelation and cross-correlation
    are to be normalised by mean and std. Default is True.

    """
    summ_stats = []
    # Mean and variance
    mean = x.mean(1).reshape(x.shape[0], 1, -1)
    var = x.var(1).reshape(x.shape[0], 1, -1) + 1e-8

    # Normalise
    if norm:
        x_norm = (x - mean) / np.sqrt(var)
    else:
        x_norm = x

    # Autocorrelation
    autocorr = np.zeros((x.shape[0], len(lags), x.shape[-1]))
    for i, l in enumerate(lags):
        autocorr[:, i, :] = (x_norm[:, l:] * x_norm[:, :-l]).mean(1)

    # Crosscorrelation
    cross_corr = np.zeros((x.shape[0], x.shape[-1], x.shape[-1]))
    for i in range(x.shape[-1]):
        for j in range(i, x.shape[-1]):
            cross_corr[:, i, j] = (x_norm[:, :, i] * x_norm[:, :, j]).mean(1)
            cross_corr[:, j, i] = cross_corr[:, i, j]
    summ_stats = np.concatenate([mean, var, autocorr, cross_corr], 1)
    return summ_stats


class SpikeNNnumpy:
    """Supralinear Spiking Neural - Hennequin et al 2018 - The Dynamical Regime of
    Sensory Cortex"""

    def __init__(
        self,
        W,
        numneur_E=2,
        numneur_I=3,
        τE=20 * 1e-3,
        τI=10 * 1e-3,
        ρ=0.2,
        dt=0.1 * 1e-3,
        τ_noise=50 * 1e-3,
        σE=0.2,
        σI=0.1,
        V0=-70.0,
        Vrest=-70.0,
        k=0.3,
        n=2,
        T=1,
    ):
        """
        :param W: connectivity matrix
        :param numneur_E: number of excitatory neurons
        :param numneur_I: number of inhibitory neurons
        :param τE: excitatory neuron time constant (in s)
        :param τI: inhibitory neuron time constant (in s)
        :param ρ: population correlation coefficient (btwn 0 and 1)
        :param dt: time-step for Euler update
        :param τ_noise: noise update time constant (in s)
        :param σE: excitatory neurons std
        :param σI: inhibitory neurons std
        :param V0: initial voltage (mV)
        :param Vrest: resting potential (mV)
        :param k: gain value for rate
        :param n: threshold rate power law
        :param T: total time of simulation
        """

        self.numneur_E, self.numneur_I = numneur_E, numneur_I
        self.num_neurs = self.numneur_E + self.numneur_I

        # set constants
        self.dt = dt
        self.τ_noise = τ_noise
        self.ρ = ρ
        self.σ = np.array([*[σE] * self.numneur_E, *[σI] * self.numneur_I])
        self.V0 = V0
        self.Vrest = Vrest
        self.k = k
        self.n = n
        self.T = T
        self.τ = np.array([*[τE] * self.numneur_E, *[τI] * self.numneur_I])

        # TODO: hard-coded for convenience -- need to make input for h more flexible,
        # but more efficient than giving it to forward()
        self.h = (
            np.array([3.3705623, 2.3463864, 2.5844836, 2.942931, 3.777861])
            .repeat(int(self.T / self.dt))
            .reshape(-1, int(self.T / self.dt))
            .T
        )

        # set parameters
        self.W = W

    @property
    def kernel(self):
        return (self.σ ** 2 * np.eye(self.num_neurs) * (1 - self.ρ) + self.ρ) / self.dt

    def rate(self, V):
        return self.k * (np.maximum(V - self.Vrest, 0) ** self.n)

    def noise(self, η):
        return -η + np.dot(
            np.sqrt(2 * self.τ_noise * self.kernel), np.random.randn(*η.shape)
        )

    def flow_eqn(self, V, r, h, η):
        η_new = η + (self.dt * self.noise(η) / self.τ_noise)
        V_new = V + (
            self.dt * (-V + self.Vrest + h + η_new + np.dot(self.W, r)) / self.τ
        )

        r_new = self.rate(V_new)

        return V_new, r_new, η_new

    def forward(self, V=None, h=None, return_summ_stats=True, save_freq=1):
        """
        Generate voltage, rate and noise traces for neural population.

        :param V_init: numpy array containing initial voltage of length = neurons. If
        None, voltage is set to V0
        :param h: external input to neurons of size timebins x neurons. If None, h is
        set to zero.
        :param return_summ_stats: boolean value. If True, returns summary statistics on
        rate traces. If False, returns voltage, rate and nosie traces.
        :param save_freq: number of update steps at which to save voltage, rate and
        noise traces. Default 1, implies values are saved after every update step.

        """
        if V is None:
            V = np.ones(self.num_neurs) * self.V0
        if h is None:
            h = self.h
        voltage = np.zeros((int(self.T / self.dt) + 1, self.num_neurs))
        rate = np.zeros_like(voltage)
        noise = np.zeros_like(voltage)

        η = np.random.randn(self.num_neurs)
        r = self.rate(V)
        voltage[0] = V
        noise[0] = η
        rate[0] = r
        for i, t in enumerate(np.arange(0, self.T, self.dt)):
            V, r, η = self.flow_eqn(V, r, h[i], η)
            if np.all(np.isfinite(V)) == False:
                print("Rates diverging to infinity.")
                voltage = np.ones_like(voltage) * np.nan
                rate = np.ones_like(voltage) * np.nan
                noise = np.ones_like(voltage) * np.nan
                break
            if i % save_freq == 0:
                voltage[i + 1] = V
                rate[i + 1] = r
                noise[i + 1] = η

        if return_summ_stats:
            return calc_summ_stats_numpy(np.expand_dims(rate, 0))
        else:
            return voltage, rate, noise


class SpikeNNPrior:
    def __init__(
        self,
        nE: int = 2,
        nI: int = 3,
        W_vals: Optional[List] = None,
        std: float = 0.09,
        bandwidth: float = 1.25,
    ):

        if W_vals is None:
            W_vals = [0.01, -0.01, 0.01, -0.01]
        else:
            assert len(W_vals) == 4

        x = np.linspace(-nE / bandwidth, nE / bandwidth, nE)
        y = np.linspace(-nI / bandwidth, nI / bandwidth, nI)

        Wee, Wei, Wie, Wii = W_vals

        # Construct mean and covariance matrix
        # Excitatory-excitatory quadrant
        X1, X2 = np.meshgrid(x, x)
        mean_EE = np.ones_like(X1) * Wee
        cov_EE = np.exp(-((X1 - X2) ** 2))

        # Excitatory-inhibitory quadrant
        Y, X = np.meshgrid(y, x)
        mean_EI = np.ones_like(X) * Wei
        cov_EI = np.exp(-((X - Y) ** 2))

        # Inhibitory-excitatory quadrant
        X, Y = np.meshgrid(x, y)
        mean_IE = np.ones_like(X) * Wie
        cov_IE = np.exp(-((X - Y) ** 2))

        # Inhibitory-inhibitory quadrant
        Y1, Y2 = np.meshgrid(y, y)
        mean_II = np.ones_like(Y1) * Wii
        cov_II = np.exp(-((Y1 - Y2) ** 2))

        mean = sl.block_diag(mean_EE, mean_II)
        mean[nE:, :nE] = mean_IE
        mean[:nE, nE:] = mean_EI

        cov = sl.block_diag(cov_EE, cov_II)
        cov[nE:, :nE] = cov_IE
        cov[:nE, nE:] = cov_EI

        self.cov = cov
        self.std = std

        self.dist = norm(loc=mean, scale=self.std * self.cov)

    def sample(self, sample_shape=torch.Size([])):

        if sample_shape == torch.Size([]):
            return torch.as_tensor(
                self.dist.rvs().reshape(self.cov.size), dtype=torch.float32
            )
        else:
            num_samples = sample_shape[0]
            W = self.dist.rvs((num_samples, *self.cov.shape))
            return torch.as_tensor(
                W.reshape(num_samples, self.cov.size), dtype=torch.float32
            )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:

        num_values, value_shape = value.shape

        value = value.detach().numpy().reshape(num_values, *self.cov.shape)
        logprobs = self.dist.logpdf(value).sum(axis=(1, 2))

        return torch.as_tensor(logprobs, dtype=torch.float32)

    @property
    def mean(self):
        return torch.as_tensor(
            self.dist.mean().reshape(self.cov.size), dtype=torch.float32
        )

    @property
    def variance(self):
        return torch.as_tensor(
            self.dist.var().reshape(self.cov.size), dtype=torch.float32
        )
