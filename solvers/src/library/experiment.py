"""Pointwise losses used by the historical no-pilot comparator loops."""
import numpy as np
from scipy.special import spence, expit
from sklearn._loss.loss import HalfPoissonLoss, HalfGammaLoss, HalfTweedieLoss

def softplus_integral(z):
    """-Li_2(-exp(z)), using reflection to avoid exponentiating positive z."""
    z = np.asarray(z)
    u = np.exp(-np.abs(z))
    dilog = spence(1 + u)
    return np.where(z > 0, .5 * np.maximum(z, 0)**2 + np.pi**2/6 + dilog,
                    -dilog)

class PointLoss:
    is_multiclass = False
    n_classes = None

    def __init__(self, tau, y, fisher=False):
        self.tau, self.delta, self.fisher = tau, None, fisher
        self.exact = (HalfPoissonLoss() if tau == 1 else HalfGammaLoss()
                      if tau == 2 else HalfTweedieLoss(power=tau))
        self.link = self.exact.link
        self.log_y = np.full_like(y, -np.inf)
        np.log(y, out=self.log_y, where=y > 0)
        self.calls = {"loss_gradient": 0, "gradient_hessian": 0, "loss": 0}

    def surrogate(self, eta, need_loss=True, observed=False):
        d, t = self.delta, self.tau
        za = self.log_y + (1-t)*eta - np.log(d)
        zb = (2-t)*eta - np.log(d)
        la, lb = np.logaddexp(0, za), np.logaddexp(0, zb)
        grad = d * (lb-la)
        curv = (d*expit(zb) if self.fisher and not observed else
                d*((t-1)*expit(za)+(2-t)*expit(zb)))
        value = None
        if need_loss:
            if t == 1:
                value = d*(softplus_integral(zb)-eta*la)
            elif t == 2:
                value = d*(eta*lb+softplus_integral(za))
            else:
                value = d*(softplus_integral(zb)/(2-t)
                           - softplus_integral(za)/(1-t))
        return value, grad, curv

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        self.calls["loss_gradient"] += 1
        if self.delta is None:
            return self.exact.loss_gradient(y_true=y_true, raw_prediction=raw_prediction,
                                           sample_weight=sample_weight, n_threads=n_threads)
        value, grad, _ = self.surrogate(raw_prediction)
        if sample_weight is not None:
            value *= sample_weight
            grad *= sample_weight
        return value, grad

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        self.calls["loss"] += 1
        if self.delta is None:
            return self.exact.loss(y_true=y_true, raw_prediction=raw_prediction,
                                   sample_weight=sample_weight, n_threads=n_threads)
        value = self.surrogate(raw_prediction)[0]
        return value if sample_weight is None else value*sample_weight

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        self.calls["gradient_hessian"] += 1
        if self.delta is None:
            grad, hess = self.exact.gradient_hessian(
                y_true=y_true, raw_prediction=raw_prediction,
                sample_weight=sample_weight, n_threads=n_threads)
            if self.fisher:
                hess = np.exp((2-self.tau)*raw_prediction)
                if sample_weight is not None:
                    hess *= sample_weight
            return grad, hess
        _, grad, hess = self.surrogate(raw_prediction, need_loss=False)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess
