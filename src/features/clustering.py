import numpy as np
import fastjet
import awkward as ak

def compute_n_subjettiness(jet, n, beta=1.0):
    """
    Computes N-subjettiness tau_N for a jet.
    Approximation using exclusive kt clustering for the N axes via fastjet.
    """
    constituents = jet.constituents()
    if len(constituents) < n:
        return 0.0
        
    # Recluster using kt to find N exclusive jets
    ktdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
    cluster = fastjet.ClusterSequence(constituents, ktdef)
    subjets = cluster.exclusive_jets(n)
    
    if len(subjets) < n:
        return 0.0
        
    # Calculate tau_N
    d0 = 0.0
    tau_n = 0.0
    R0 = 1.0 
    
    for c in constituents:
        pt_c = c.pt
        eta_c = c.eta
        phi_c = c.phi
        
        d0 += pt_c * (R0 ** beta)
        
        min_dist = float('inf')
        for sj in subjets:
            deta = eta_c - sj.eta
            dphi = phi_c - sj.phi
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            dR = np.sqrt(deta**2 + dphi**2)
            if dR < min_dist:
                min_dist = dR
                
        tau_n += pt_c * (min_dist ** beta)
        
    if d0 == 0:
        return 0.0
        
    return tau_n / d0

def extract_event_features(event_row):
    """
    Extracts features for a single event.
    event_row: array of shape (2100,) containing pt, eta, phi for 700 particles.
    """
    particles = event_row.reshape(-1, 3)
    mask = particles[:, 0] > 0
    valid_particles = particles[mask]
    
    if len(valid_particles) == 0:
        return None
        
    # Create fastjet PseudoJets
    pts = valid_particles[:, 0]
    etas = valid_particles[:, 1]
    phis = valid_particles[:, 2]
    # massless particles
    ms = np.zeros_like(pts)
    
    # fastjet expects px, py, pz, E or awkward array of pt, eta, phi, mass
    # Use awkward array
    array = ak.zip({
        "pt": pts,
        "eta": etas,
        "phi": phis,
        "mass": ms
    })
    
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
    cluster = fastjet.ClusterSequence(array, jetdef)
    jets = cluster.inclusive_jets(ptmin=0.0)
    
    # Sort jets by pt
    jets_sorted = sorted(jets, key=lambda j: j.pt, reverse=True)
    
    if len(jets_sorted) < 2:
        return None
        
    j1 = jets_sorted[0]
    j2 = jets_sorted[1]
    
    mJ1 = j1.m
    mJ2 = j2.m
    dmJ = abs(mJ1 - mJ2)
    
    tau1_j1 = compute_n_subjettiness(j1, 1)
    tau2_j1 = compute_n_subjettiness(j1, 2)
    tau21_j1 = tau2_j1 / tau1_j1 if tau1_j1 > 0 else 0.0
    
    tau1_j2 = compute_n_subjettiness(j2, 1)
    tau2_j2 = compute_n_subjettiness(j2, 2)
    tau21_j2 = tau2_j2 / tau1_j2 if tau1_j2 > 0 else 0.0
    
    deta = j1.eta - j2.eta
    dphi = j1.phi - j2.phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dRJJ = np.sqrt(deta**2 + dphi**2)
    
    p1 = np.array([j1.e, j1.px, j1.py, j1.pz])
    p2 = np.array([j2.e, j2.px, j2.py, j2.pz])
    p_tot = p1 + p2
    
    mJJ2 = p_tot[0]**2 - p_tot[1]**2 - p_tot[2]**2 - p_tot[3]**2
    mJJ = np.sqrt(mJJ2) if mJJ2 > 0 else 0.0
    
    return {
        'mJ1': float(mJ1),
        'dmJ': float(dmJ),
        'tau21_J1': float(tau21_j1),
        'tau21_J2': float(tau21_j2),
        'dRJJ': float(dRJJ),
        'mJJ': float(mJJ)
    }
