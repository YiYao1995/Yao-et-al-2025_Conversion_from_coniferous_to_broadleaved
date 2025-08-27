from scipy import stats
import numpy as np
import xarray as xr
import statsmodels as sm
import statsmodels.stats.multitest

# Significance testing on gridded data (xarray)

# Student t-test: parametric test for independent/dependent samples, data is normally distributed 
def ttest_2samp(da1, da2, paired=False, dim="time", global_alpha=0.05):
    """xarray wrapper for two-sample student's t test
    Paramters
    ---------
    da1, da2 : xr.DataArray
        arrays of samples
    paired : bool, default: False
        T-test unpaired or paired (identical to 1samp of difference, for correlated samples)
    dim : str, default: "time"
        Dimension along which to compute the test
    global_alpha, float, default: 0.05
        Global alpha of Benjamini and Hochberg correction, 2x the level for non-field significance tests in Wilks (2016)
        For two-tailed tests the non-field significance level has to be halved, which cancels the global significance multiplier
    
    """
    dim = [dim] if isinstance(dim, str) else dim

    def _ttest(x, y):
        if paired: # if True
            return stats.ttest_rel(x, y, axis=-1, nan_policy="omit").pvalue # paired (related samples); default is two-tailed
        else:
            return stats.ttest_ind(x, y, axis=-1, nan_policy="omit").pvalue # unpaired; default is two-tailed

    # use xr.apply_ufunc to handle vectorization
    result = xr.apply_ufunc(
        _ttest,
        da1,
        da2,
        input_core_dims=[dim, dim],
        output_core_dims=[[]],
        exclude_dims=set(dim),
        dask="parallelized",
        output_dtypes=[float],
    ).compute()

    # apply Benjamini and Hochberg correction
    shape = result.shape
    
    values = result.values.ravel()
    notnull = np.isfinite(values)

    p_adjust = sm.stats.multitest.multipletests(
        values[notnull], alpha=global_alpha, method="fdr_bh"
    )[0]
    values[notnull] = p_adjust
    result.values = values.reshape(shape)

    return result

# Mann-Whitney U rank test: non-parametric test for independent samples
# Wilcoxon rank-sum test: non-parametric test for dependent samples
# Equivalent to two-sample t test if the data is not normally distributed and ordinal; also applicable for small samples and outliers.
def mannwhitney(da1, da2, paired=False, dim="time", global_alpha=0.05):
    """xarray wrapper for Mann-Whitney U test
    Paramters
    ---------
    da1, da2 : xr.DataArray
        arrays of samples
    dim : str, default: "time"
        Dimension along which to compute the test
    global_alpha, float, default: 0.05
        Global alpha of Benjamini and Hochberg correction   
    
    """
    dim = [dim] if isinstance(dim, str) else dim

    def _mannwhitneyu(x, y):
        if paired: # if True
            return scipy.stats.wilcoxon(x, y, axis=-1, nan_policy="omit").pvalue
        else:
            return stats.mannwhitneyu(x, y, axis=-1, nan_policy="omit").pvalue

    # use xr.apply_ufunc to handle vectorization
    result = xr.apply_ufunc(
        _mannwhitneyu,
        da1,
        da2,
        input_core_dims=[dim, dim],
        output_core_dims=[[]],
        exclude_dims=set(dim),
        dask="parallelized",
        output_dtypes=[float],
    ).compute()

    # apply Benjamini and Hochberg correction
    shape = result.shape
    
    values = result.values.ravel()
    notnull = np.isfinite(values)

    p_adjust = sm.stats.multitest.multipletests(
        values[notnull], alpha=global_alpha, method="fdr_bh"
    )[0]
    values[notnull] = p_adjust
    result.values = values.reshape(shape)

    return result