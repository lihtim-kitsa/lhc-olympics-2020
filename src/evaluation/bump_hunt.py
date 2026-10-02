"""Fixed-window sideband fit with a Poisson likelihood and fit uncertainty."""
import numpy as np
from scipy.optimize import minimize


class BumpHunter:
    def __init__(self, fit_range=(2500,4500), signal_window=(3300,3700), n_bins=40):
        self.fit_range=fit_range; self.signal_window=signal_window; self.n_bins=n_bins
        self.bins=np.linspace(*fit_range,n_bins+1)
        self.centers=.5*(self.bins[:-1]+self.bins[1:])
        self.sideband=(self.centers<signal_window[0])|(self.centers>signal_window[1])
        self.signal=~self.sideband

    @staticmethod
    def basis(mass):
        x=np.clip(np.asarray(mass,dtype=float)/13000.,1e-8,1-1e-8)
        lx=np.log(x)
        return np.column_stack([np.ones_like(x),np.log1p(-x),-lx,-lx*lx])

    def fit_background(self,mjj_data,weights=None):
        counts,_=np.histogram(np.asarray(mjj_data,dtype=float),bins=self.bins,weights=weights)
        y=counts[self.sideband].astype(float); design=self.basis(self.centers[self.sideband])
        positive=y>0
        if positive.sum()<design.shape[1]+2: return np.nan,np.nan,np.nan,np.nan
        # Log-linear initialization followed by an extended binned Poisson fit.
        theta=np.linalg.lstsq(design[positive],np.log(y[positive]),rcond=None)[0]
        def objective(q):
            logmu=np.clip(design@q,-50,50); mu=np.exp(logmu)
            return float(np.sum(mu-y*logmu))
        def gradient(q):
            logmu=np.clip(design@q,-50,50); mu=np.exp(logmu)
            return design.T@(mu-y)
        bounds=[(-50,50),(-20,60),(-20,60),(-20,20)]
        fit=minimize(objective,theta,jac=gradient,bounds=bounds,method='L-BFGS-B',options={'maxiter':5000,'ftol':1e-12})
        # A finite parameter vector from a failed fit is not a valid closure
        # result; accepting it can create absurd expected yields and Z values.
        if not fit.success or not np.isfinite(fit.fun): return np.nan,np.nan,np.nan,np.nan
        theta=fit.x
        sideband_expected=np.exp(np.clip(design@theta,-50,50))
        observed_sideband=float(y.sum())
        fitted_sideband=float(sideband_expected.sum())
        # Reject numerically converged but saturated fits whose sideband yield
        # is inconsistent with the observed sidebands; these otherwise predict
        # ~0 events in the window and produce spurious enormous local Z values.
        if (observed_sideband <= 0 or not np.isfinite(fitted_sideband)
                or fitted_sideband < 0.5 * observed_sideband
                or fitted_sideband > 2.0 * observed_sideband):
            return np.nan,np.nan,np.nan,np.nan
        design_sig=self.basis(self.centers[self.signal])
        mu_sig=np.exp(np.clip(design_sig@theta,-50,50))
        b=float(mu_sig.sum()); n=float(counts[self.signal].sum())
        design_fit=self.basis(self.centers[self.sideband])
        mu_fit=np.exp(np.clip(design_fit@theta,-50,50))
        fisher=design_fit.T@(mu_fit[:,None]*design_fit)
        cov=np.linalg.pinv(fisher,rcond=1e-12)
        grad_b=design_sig.T@mu_sig
        var_b=float(grad_b@cov@grad_b)
        sigma=float(np.sqrt(max(0,var_b))) if np.isfinite(var_b) else np.nan
        z=0.0
        if n>b and b>0 and np.isfinite(sigma):
            s2=sigma*sigma
            if s2<1e-12: z=float(np.sqrt(2*(n*np.log(n/b)-(n-b))))
            else:
                t1=n*np.log((n*(b+s2))/(b*b+n*s2))
                t2=(b*b/s2)*np.log(1+(s2*(n-b))/(b*(b+s2)))
                z=float(np.sqrt(max(0,2*(t1-t2))))
        return n,b,z,sigma


def perform_bump_hunt(mjj_data,prefix='',weights=None):
    observed,expected,significance,uncertainty=BumpHunter().fit_background(mjj_data,weights=weights)
    return {f'{prefix}observed_events':float(observed),f'{prefix}expected_bkg':float(expected),
        f'{prefix}local_significance':float(significance),f'{prefix}expected_bkg_uncertainty':float(uncertainty)}


def bootstrap_null_significances(mjj_data,n_trials=50,seed=42,observed_z=None,base_weights=None):
    """Poisson bootstrap the background-only events and refit each pseudoexperiment."""
    data=np.asarray(mjj_data,dtype=float)
    base=np.ones(len(data),dtype=float) if base_weights is None else np.asarray(base_weights,dtype=float)
    if base.shape != data.shape or np.any(~np.isfinite(base)) or np.any(base < 0):
        raise ValueError('base_weights must be finite, non-negative, and align with mjj_data')
    rng=np.random.default_rng(seed); values=[]
    hunter=BumpHunter()
    for _ in range(n_trials):
        # Fractional base weights represent randomized acceptance of tied
        # boundary scores. Draw that Bernoulli decision per pseudoexperiment
        # before Poisson event multiplicities to preserve its variance.
        acceptance = rng.binomial(1, np.clip(base, 0., 1.))
        weights=rng.poisson(1.0,size=len(data))*acceptance
        _,_,z,_=hunter.fit_background(data,weights=weights)
        if np.isfinite(z): values.append(float(z))
    if not values: return {'mean':np.nan,'median':np.nan,'q95':np.nan,'n':0,'pvalue':np.nan}
    pvalue=(1+sum(v>=observed_z for v in values))/(len(values)+1) if observed_z is not None else np.nan
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),
        'q95':float(np.quantile(values,.95)),'n':len(values),'pvalue':float(pvalue)}
