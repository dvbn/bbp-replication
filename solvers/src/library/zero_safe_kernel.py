"""Exact zero-row Gamma/Tweedie loss extension used by the study."""
import numpy as np
import engine
OriginalCountedPointLoss = engine.CountedPointLoss

class StableCountedPointLoss(OriginalCountedPointLoss):
    def __init__(self, tau, y, fisher, maxfun):
        super().__init__(tau, y, fisher, maxfun)
        self.zero_rows = np.asarray(y) == 0
        self.patch_zeros = bool(tau > 1 and np.any(self.zero_rows))

    def _zero_contributions(self, raw_prediction, sample_weight):
        eta = raw_prediction[self.zero_rows]
        if self.tau == 2:
            loss, gradient = eta.copy(), np.ones_like(eta)
            hessian = np.ones_like(eta) if self.fisher else np.zeros_like(eta)
        else:
            with np.errstate(over="ignore", under="ignore"):
                gradient = np.exp((2-self.tau)*eta)
            loss = gradient/(2-self.tau)
            hessian = gradient if self.fisher else (2-self.tau)*gradient
        if sample_weight is not None:
            weight = np.asarray(sample_weight)[self.zero_rows]
            loss, gradient, hessian = loss*weight, gradient*weight, hessian*weight
        return loss, gradient, hessian

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        value = super().loss(y_true, raw_prediction, sample_weight, n_threads)
        if self.patch_zeros and self.delta is None:
            value[self.zero_rows] = self._zero_contributions(raw_prediction, sample_weight)[0]
        return value

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        value, gradient = super().loss_gradient(y_true, raw_prediction, sample_weight, n_threads)
        if self.patch_zeros and self.delta is None:
            zero_loss, zero_gradient, _ = self._zero_contributions(raw_prediction, sample_weight)
            value[self.zero_rows], gradient[self.zero_rows] = zero_loss, zero_gradient
        return value, gradient

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        gradient, hessian = super().gradient_hessian(y_true, raw_prediction, sample_weight, n_threads)
        if self.patch_zeros and self.delta is None:
            _, zero_gradient, zero_hessian = self._zero_contributions(raw_prediction, sample_weight)
            gradient[self.zero_rows], hessian[self.zero_rows] = zero_gradient, zero_hessian
        return gradient, hessian

def install():
    """Explicitly select the stable exact zero-row extension in this process."""
    engine.CountedPointLoss = StableCountedPointLoss
