import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon

def calculate_rejection_at_efficiency(y_true, y_score, target_effs=[0.01, 0.05, 0.10, 0.30, 0.50]):
    """
    Calculate background rejection (1 / eB) at given signal efficiencies (eS).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # tpr is signal efficiency, fpr is background efficiency
    
    results = {}
    for eff in target_effs:
        # Find index where tpr is closest to and >= target_eff
        idx = np.where(tpr >= eff)[0]
        if len(idx) > 0:
            idx = idx[0]
            eB = fpr[idx]
            rej = 1.0 / eB if eB > 0 else np.inf
            results[f'rej_{int(eff*100)}'] = float(rej)
        else:
            results[f'rej_{int(eff*100)}'] = np.nan
            
    return results

def calculate_max_sic(y_true, y_score):
    """
    Calculate Maximum Significance Improvement Characteristic (SIC)
    SIC = eS / sqrt(eB)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Avoid division by zero
    valid = fpr > 0
    tpr_v = tpr[valid]
    fpr_v = fpr[valid]
    thresh_v = thresholds[valid]
    
    if len(tpr_v) == 0:
        return {'max_sic': np.nan, 'max_sic_threshold': np.nan}
        
    sic = tpr_v / np.sqrt(fpr_v)
    max_idx = np.argmax(sic)
    
    return {
        'max_sic': float(sic[max_idx]),
        'max_sic_threshold': float(thresh_v[max_idx])
    }

def calculate_js_divergence(hist1, hist2):
    """
    Calculate Jensen-Shannon Divergence between two histograms.
    hist1, hist2 should be normalized (sum to 1).
    """
    # JS distance is returned, divergence is distance squared
    js_dist = jensenshannon(hist1, hist2)
    return float(js_dist ** 2)

def evaluate_mass_sculpting(mjj_bkg, score_bkg, thresholds):
    """
    Evaluate JS divergence of mJJ background before and after cuts.
    thresholds: dict like {'10pct': t10, '1pct': t1}
    mjj_bkg: mjj values for background events
    score_bkg: anomaly scores for background events
    """
    # Define bins
    bins = np.linspace(2000, 5000, 50)
    
    hist_incl, _ = np.histogram(mjj_bkg, bins=bins, density=True)
    hist_incl = hist_incl * np.diff(bins) # normalize to sum 1
    
    results = {}
    for name, thresh in thresholds.items():
        # Apply cut: score >= thresh
        selected_mjj = mjj_bkg[score_bkg >= thresh]
        if len(selected_mjj) > 0:
            hist_cut, _ = np.histogram(selected_mjj, bins=bins, density=True)
            hist_cut = hist_cut * np.diff(bins)
            
            jsd = calculate_js_divergence(hist_incl, hist_cut)
            results[f'mass_sculpting_jsd_at_{name}_bkg'] = jsd
        else:
            results[f'mass_sculpting_jsd_at_{name}_bkg'] = np.nan
            
    return results

def calculate_score_mjj_dependence(mjj_bkg, score_bkg):
    """
    Calculate Pearson correlation coefficient between anomaly score and mJJ on background.
    """
    if len(mjj_bkg) < 2:
        return 0.0
    corr = np.corrcoef(mjj_bkg, score_bkg)[0, 1]
    return float(corr)

def get_core_metrics(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    metrics = {'roc_auc': float(roc_auc)}
    metrics.update(calculate_rejection_at_efficiency(y_true, y_score))
    metrics.update(calculate_max_sic(y_true, y_score))
    
    return metrics
