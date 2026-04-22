import numpy as np
import scipy.stats


def calc_ccf(local_cn_a1, local_cn_a2, alt_cnt, ref_cnt, purity, grid_size=101):
    """
    Calculate CCF from local copy number, alt/ref count, and purity
    Args:
        local_cn_a1: Minor allele local copy number
        local_cn_a2: Major allele local copy number
        alt_cnt: Alt count
        ref_cnt: Ref count
        purity: Tumor fraction
        grid_size: number of bins

    Returns:
        numpy array representing the CCF histogram
    """
    # Separating clonal and subclonal copy number components for each allele
    
    # If local CN is > 1, then the clonal CN is the floor (lower integer) and the subclonal CN is the ceiling (upper integer), 
    # with the subclonal fraction being the decimal remainder. 
    print("local_cn_a1:", local_cn_a1, "local_cn_a2:", local_cn_a2)
    if local_cn_a1 > 1.:
        clonal_cn_a1 = np.floor(local_cn_a1)
        subclonal_cn_a1 = np.ceil(local_cn_a1)
        subclonal_frac_a1 = local_cn_a1 - clonal_cn_a1
    # If local CN is < 1, then the clonal CN is the ceiling (upper integer) and the subclonal CN is the floor (lower integer), 
    # with the subclonal fraction being the decimal remainder.
    else:
        clonal_cn_a1 = np.ceil(local_cn_a1)
        subclonal_cn_a1 = np.floor(local_cn_a1)
        subclonal_frac_a1 = clonal_cn_a1 - local_cn_a1

    
    if local_cn_a2 > 1.:
        clonal_cn_a2 = np.floor(local_cn_a2)
        subclonal_cn_a2 = np.ceil(local_cn_a2)
        subclonal_frac_a2 = local_cn_a2 - clonal_cn_a2
    else:
        clonal_cn_a2 = np.ceil(local_cn_a2)
        subclonal_cn_a2 = np.floor(local_cn_a2)
        subclonal_frac_a2 = clonal_cn_a2 - local_cn_a2

    total_cov = alt_cnt + ref_cnt
    total_cn = local_cn_a1 + local_cn_a2

    # Calculate likelihood of clonality and subclonality

    # Likelihood of multiplicity = 1 : probability of being subclonal, because amplifications are often early events, 
    # so if mutation not amplified, it probably arrived later and would be subclonal
    # Distribution of CCF because could also be clonal if diploid region, or mutation on parental chromosome that wasn't amplified
    # CCF distribution for m=1
    ccf_dist_m1 = ccf_dist_from_params(1, total_cn, alt_cnt, ref_cnt, purity, grid_size=grid_size)
    # extract CCF with the highest score (ccf = indices of scores)
    ccf_mode = np.argmax(ccf_dist_m1) / 100.
    # compute VAF for this CCF
    af_mode = ccf_mode * purity / (total_cn * purity + 2 * (1 - purity))
    # compute likelihood of observing alt count muted reads for total_cov reads given this VAF under binomial model
    p_subclonal = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
    p_clonal = 0

    # Add likelihood of each integer multiplicity : probability of being clonal because appeared before amplification
    for mult in np.arange(2, clonal_cn_a2 + 1):
        af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
        mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
        p_clonal += mult_weight

    # Add likelihood of subclonal shifts in multiplicity > 1 due to gains/deletions
    # Consider possibility of m>1 and still subclonal : mutation appeared before amplification but only on a subset of cells
    # Considered only if subclonal_frac != 0 (local_cn are not both integers), so amplification not in all tumoral cells
    if subclonal_frac_a1:
        for mult in np.arange(subclonal_frac_a1 + 1, subclonal_cn_a1):
            af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
            mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
            p_clonal += mult_weight
    if subclonal_frac_a2:
        for mult in np.arange(subclonal_frac_a2 + 1, subclonal_cn_a2):
            af_mode = mult * purity / (total_cn * purity + 2 * (1 - purity))
            mult_weight = scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode)
            p_clonal += mult_weight

    # Calculate ccf for multiplicity 1
    subc_ccf_hist = np.zeros(grid_size)
    if local_cn_a2 >= 1.:
        # Add mult1 ccf dist weighted by maximum subclonal likelihood
        subc_ccf_hist += ccf_dist_m1 * p_subclonal
    # If subclonality is from subclonal deletion, cut off CCF at subclonal fraction and weight by CCF likelihood
    # Two hypothesis : subclonal fraction on one cell population (dist1) or on the other (dist2)
    if subclonal_frac_a1:
        subc_ccf_dist1 = ccf_dist_m1.copy()
        subc_ccf_dist2 = ccf_dist_m1.copy()
        subc_ccf_dist1[int(subclonal_frac_a1 * 100):] = 0
        subc_ccf_dist2[int((1 - subclonal_frac_a1) * 100):] = 0
        sum_dist_1 = sum(subc_ccf_dist1)
        if sum_dist_1:
            subc_ccf_dist1 /= sum_dist_1
            af_mode_1 = subclonal_frac_a1 * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist1 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_1)
        sum_dist_2 = sum(subc_ccf_dist2)
        if sum_dist_2:
            subc_ccf_dist2 /= sum_dist_2
            af_mode_2 = (1 - subclonal_frac_a1) * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist2 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_2)
    if subclonal_frac_a2:
        subc_ccf_dist1 = ccf_dist_m1.copy()
        subc_ccf_dist2 = ccf_dist_m1.copy()
        subc_ccf_dist1[int(subclonal_frac_a2 * 100):] = 0
        subc_ccf_dist2[int((1 - subclonal_frac_a2) * 100)] = 0
        sum_dist_1 = sum(subc_ccf_dist1)
        if sum_dist_1:
            subc_ccf_dist1 /= sum_dist_1
            af_mode_1 = subclonal_frac_a2 * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist1 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_1)
        sum_dist_2 = sum(subc_ccf_dist2)
        if sum_dist_2:
            subc_ccf_dist2 /= sum_dist_2
            af_mode_2 = (1 - subclonal_frac_a2) * purity / (total_cn * purity + 2 * (1 - purity))
            subc_ccf_hist += subc_ccf_dist2 * scipy.stats.binom.pmf(alt_cnt, total_cov, af_mode_2)
    subc_ccf_hist /= sum(subc_ccf_hist)

    # Create a typical clonal histogram, with peak at CCF = 1
    bins = np.linspace(0, 1, grid_size)
    clonal_ccf_hist = scipy.stats.beta.pdf(bins, alt_cnt + 1, 1)
    clonal_ccf_hist /= sum(clonal_ccf_hist)

    # Add clonal and subclonal histograms weighted by their respective likelihoods, and normalize
    ccf_hist = p_subclonal * subc_ccf_hist + p_clonal * clonal_ccf_hist
    ccf_hist /= sum(ccf_hist)

    return ccf_hist


def ccf_dist_from_params(mult, total_cn, alt_cnt, ref_cnt, purity, grid_size=101):
    """
    Calculate a ccf distribution for a given multiplicity
    Args:
        mult: multiplicity
        total_cn: total local copy number
        alt_cnt: alt count
        ref_cnt: ref count
        purity: tumor fraction
        grid_size: number of bins

    Returns:
        CCF distribution and mean ccf for given multiplicity
    """
    if mult == 0:
        ccf_dist = np.zeros(grid_size)
        ccf_dist[0] = 1.
        return ccf_dist, 0.

    ccf_bins = np.linspace(0, 1, grid_size)

    # Since transformation is linear, ToV formula not necessary
    # compute variant allele frequency, for each possible value of CCF in ccf_bins
    ccf_domain_in_af_space = ccf_bins * mult * purity / (total_cn * purity + 2 * (1 - purity))
    # probability density (Beta law) of observing these VAFs given alt/ref counts
    # if high alt/ref counts, the distribution will be more peaked around the mean
    # if low alt/ref counts, the distribution will be more spread out
    # +1 as Laplace smoothing to avoid zero probabilities when alt_cnt or ref_cnt is 0
    ccf_dist = scipy.stats.beta.pdf(ccf_domain_in_af_space, alt_cnt + 1, ref_cnt + 1)
    # Normalize distribution to sum to 1
    return ccf_dist / sum(ccf_dist)
