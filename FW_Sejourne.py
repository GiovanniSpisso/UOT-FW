import numpy as np
from tqdm import tqdm
from fastuot.uot1d import solve_ot, lazy_potential

def kl_divergence(mu, nu, eps=1e-300):
    """KL(mu || nu) = sum_i [ mu_i * log(mu_i/nu_i) - mu_i + nu_i ].
    Handle zeros robustly: if mu_i == 0 then contribution is nu_i (since 0*log = 0 and -0)
    If nu_i == 0 and mu_i > 0 -> return +inf (we avoid by adding tiny eps to nu if needed).
    eps is tiny regularizer to avoid log(0)."""
    mu = np.asarray(mu, dtype=float)
    nu = np.asarray(nu, dtype=float)

    # If any nu==0 while mu>0, KL = +inf (strict definition). We handle by raising.
    mask_bad = (nu <= 0) & (mu > 0)
    if np.any(mask_bad):
        return np.inf

    # safe division: add tiny eps only where mu>0 to avoid log(0)/0 warnings for mu==0
    # but do not change the strict check above for nu==0 & mu>0
    safe_nu = nu.copy()
    safe_nu[safe_nu <= 0] = eps

    # compute terms: for mu>0: mu*log(mu/nu) - mu + nu; for mu==0: term = nu
    result = 0.0
    pos = mu > 0
    if np.any(pos):
        result += np.sum(mu[pos] * np.log(mu[pos] / safe_nu[pos]) - mu[pos] + nu[pos])
    if np.any(~pos):
        result += np.sum(nu[~pos])  # because mu=0 => term reduces to +nu

    return result


def primal_uot_value_from_atoms(I, J, P, x, y, a, b, p=1, rho1=1.0, rho2=None):
    """Compute primal UOT (KL case) value from atoms I,J,P (I,J are integer arrays, P are masses).
    x,y are supports (used only for cost), a,b are original histograms (weights).
    p is cost exponent. rho1,rho2 are KL penalty weights. If rho2 is None set to rho1.
    """
    if rho2 is None:
        rho2 = rho1

    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)
    P = np.asarray(P, dtype=float)

    # cost term: sum_k P_k * |x[I_k] - y[J_k]|^p
    costs = np.abs(x[I] - y[J]) ** p
    transport_cost = np.sum(P * costs)

    # compute marginals of the plan
    # pi1[i] = sum_{k: I_k == i} P_k
    # pi2[j] = sum_{k: J_k == j} P_k
    pi1 = np.zeros_like(a, dtype=float)
    for idx, mass in zip(I, P):
        pi1[idx] += mass
    pi2 = np.zeros_like(b, dtype=float)
    for jdx, mass in zip(J, P):
        pi2[jdx] += mass

    # KL terms (use definition in the paper)
    kl1 = rho1 * kl_divergence(pi1, a)
    kl2 = rho2 * kl_divergence(pi2, b)

    if np.isinf(kl1) or np.isinf(kl2):
        return np.inf

    primal_value = transport_cost + kl1 + kl2
    return primal_value


def init_greed_uot_pers(a, b, x, y, p, rho1, rho2=None):
    if rho2 is None:
        rho2 = rho1

    _, _, _, fb, gb, _ = solve_ot(a / np.sum(a), b / np.sum(b), x, y, p)
    fc, gc = lazy_potential(x, y, p)

    # Output best convex combination
    #t = homogeneous_line_search(fb, gb, fc - fb, gc - gb, a, b, rho1, rho2,
    #                            nits=3)
    t = 0.1
    ft = (1 - t) * fb + t * fc
    gt = (1 - t) * gb + t * gc
    return ft, gt


def solve_uot_with_cost_tracking(a, b, x, y, p, rho1, rho2=None, niter=100, tol=1e-10,
                                  greed_init=False, line_search='default', stable_lse=True):
    """
    Manual implementation of solve_uot that tracks primal costs and gaps at each iteration.
    Implements the same algorithm as uot1d.solve_uot but tracks costs and gaps along the way.
    """
    from fastuot.uot1d import (solve_ot, rescale_potentials, logsumexp,
                               invariant_dual_loss, primal_dual_gap,
                               init_greed_uot, homogeneous_line_search,
                               newton_line_search)
    
    assert line_search in ['homogeneous', 'newton', 'default']
    if rho2 is None:
        rho2 = rho1

    # Initialize potentials
    if greed_init:
        #f, g = init_greed_uot(x_marg, x_marg, x, y, p, rho1, rho2)
        f, g = init_greed_uot(a, b, x, y, p, rho1, rho2)
    else:
        f, g = np.zeros_like(a), np.zeros_like(b)

    costs_per_iter = []
    primal_gaps = []
    dual_gaps = []
    
    for k in tqdm(range(niter)):
        # Output FW descent direction
        tau = (rho1 * rho2) / (rho1 + rho2)
        transl = tau * (logsumexp(-f / rho1, a, stable_lse=stable_lse) -
                        logsumexp(-g / rho2, b, stable_lse=stable_lse))
        f, g = f + transl, g - transl
        
        A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
        I, J, P, fd, gd, cost = solve_ot(A, B, x, y, p)

        # Line search - convex update
        if line_search == 'homogeneous':
            t = homogeneous_line_search(f, g, fd - f, gd - g,
                                        a, b, rho1, rho2, nits=5)
        elif line_search == 'newton':
            t = newton_line_search(f, g, fd - f, gd - g,
                                   a, b, rho1, rho2, nits=5)
        else:  # default
            t = 2. / (2. + k)
            
        f = f + t * (fd - f)
        g = g + t * (gd - g)

        # Compute primal cost at this iteration (using primal_uot_value_from_atoms)
        primal_gap = primal_uot_value_from_atoms(I, J, P, x, y, a, b, p, rho1, rho2)
        costs_per_iter.append(primal_gap)
        
        # Compute dual loss
        dual_gap = invariant_dual_loss(f, g, a, b, rho1, rho2)
        
        primal_gaps.append(primal_gap)
        dual_gaps.append(dual_gap)
        
        pdg = primal_dual_gap(a, b, x, y, p, f, g, P, I, J, rho1, rho2=None)
        if pdg < tol:
            break

    # Last iteration
    tau = (rho1 * rho2) / (rho1 + rho2)
    transl = tau * (logsumexp(-f / rho1, a, stable_lse=stable_lse) -
                    logsumexp(-g / rho2, b, stable_lse=stable_lse))
    f, g = f + transl, g - transl
    A, B = a * np.exp(-f / rho1), b * np.exp(-g / rho2)
    I, J, P, _, _, cost = solve_ot(A, B, x, y, p)
    
    # Final primal cost (using primal_uot_value_from_atoms)
    primal_gap_final = primal_uot_value_from_atoms(I, J, P, x, y, a, b, p, rho1, rho2)
    costs_per_iter.append(primal_gap_final)
    
    # Final dual loss
    dual_gap_final = invariant_dual_loss(f, g, a, b, rho1, rho2)
    
    primal_gaps.append(primal_gap_final)
    dual_gaps.append(dual_gap_final)
    
    return I, J, P, f, g, cost, costs_per_iter, primal_gaps, dual_gaps