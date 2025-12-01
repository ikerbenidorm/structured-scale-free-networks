import numpy as np
from scipy import stats

def calculate_ccdf(degree_dist):
    sorted_degrees = sorted(degree_dist.keys())
    x = np.array(sorted_degrees)
    
    pdf = np.array([degree_dist[k] for k in sorted_degrees])
  
    ccdf = np.cumsum(pdf[::-1])[::-1]
    
    return x, ccdf

def power_law_fit(x_data, y_data, k_min=None, k_max=None):

    lower_bound = k_min if k_min is not None else x_data[0]
    upper_bound = k_max if k_max is not None else x_data[-1]

    mask = (x_data >= lower_bound) & (x_data <= upper_bound)
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    if len(x_fit) < 2:
        return 0, 0, x_data, y_data

    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    r_squared = r_value**2
    
    y_fit_line = (10**intercept) * (x_data**slope)

    return slope, r_squared, x_data, y_fit_line