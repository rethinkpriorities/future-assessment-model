import time

import numpy as np
import matplotlib.pyplot as plt
import squigglepy as sq


def numerize(num):
    scales = {'thousand': 1000,
              'million': 10 ** 6,
              'billion': 10 ** 9,
              'trillion': 10 ** 12,
              'quadrillion': 10 ** 15,
              'quintillion': 10 ** 18,
              'sextillion': 10 ** 21,
              'septillion': 10 ** 24,
              'octillion': 10 ** 27,
              'nonillion': 10 ** 30,
              'decillion': 10 ** 33}

    if num < 1000:
        return num

    for scale_name, scale_value in scales.items():
        if num < scale_value * 1000:
            return str(int(round(num / scale_value))) + ' ' + scale_name

    return str(numerize(num / 10 ** 33)) + ' decillion'


def format_gb(gb):
    if gb >= 1000:
        tb = np.round(gb / 1000)
    else:
        return str(gb) + ' GB'

    if tb >= 1000:
        pb = np.round(tb / 1000)
    else:
        return str(tb) + ' TB'

    if pb >= 10000:
        return numerize(math.log10(pb)) + ' PB'
    else:
        return str(pb) + ' PB'


def generalized_logistic_curve(x, slope, shift, push, maximum, minimum):
     return minimum + ((maximum - minimum) / ((1 + shift * math.exp(-slope * x)) ** (1/push)))


def plot_tai(plt, years, cost_of_tai_collector, willingness_collector):
    cost = np.log10(np.array(cost_of_tai_collector))
    willingness = np.log10(np.array(willingness_collector))
    plt.plot(years[:len(cost)], cost, label='Cost of TAI')
    plt.plot(years[:len(willingness)], willingness, label='Willingness to pay for TAI')
    plt.legend()
    plt.ylabel('log $')
    return plt


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def log_flop_to_petaflop_sdays(log_flop):
    return round((10 ** log_flop) / (8.64 * (10 ** 19)))


def _mark_time(start, expected_sec=None, label=None,
               tolerance_ratio=1.05, tolerance_ms_threshold=5):
    end = time.time()
    delta_sec = end - start
    use_delta = delta_sec
    expected = expected_sec
    delta_label = 'sec'
    if delta_sec < 1:
        delta_ms = delta_sec * 1000
        expected = expected_sec * 1000 if expected_sec is not None else None
        use_delta = delta_ms
        delta_label = 'ms'
    use_delta = round(use_delta, 2)
    out = '...{} in {}{}'.format(label, use_delta, delta_label)
    if expected_sec is not None:
        out += ' (expected ~{}{})'.format(expected, delta_label)
    print(out)

    deviation = None
    if expected is not None:
        if delta_label == 'ms':
            deviation = not _within(use_delta, expected, tolerance_ratio, tolerance_ms_threshold)
        else:
            deviation = not _within(use_delta, expected, tolerance_ratio)
        if deviation:
            print('!!! WARNING: Unexpected timing deviation')

    return {'timing(sec)': delta_sec, 'deviation': deviation}