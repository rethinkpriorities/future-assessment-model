import time

import numpy as np
import matplotlib.pyplot as plt
import squigglepy as sq


def numerize(num, digits=1):
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

    if num < -1000:
        return '-{}'.format(numerize(-num))

    if num < 1000:
        return num

    for scale_name, scale_value in scales.items():
        if num < scale_value * 1000:
            if digits == 0:
                return str(int(round(num / scale_value))) + ' ' + scale_name
            else:
                return str(round(num / scale_value, digits)) + ' ' + scale_name

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
     return minimum + ((maximum - minimum) / ((1 + shift * math.exp(-slope * x)) ** (1 / push)))


def derive_nonscaling_delay_curve(nonscaling_points, verbose=True):
    if isinstance(nonscaling_points, dict):
        init_year = nonscaling_points['init']
        nonscaling_points = nonscaling_points['points']
    else:
        init_year = CURRENT_YEAR

    years = list(range(CURRENT_YEAR, nonscaling_points[-1][0]))
    year_cuts = [y[0] for y in nonscaling_points]
    desired = [d[1] for d in nonscaling_points]
    minimum = desired[0]
    maximum = desired[-1]
    bottom_year = year_cuts[-1]
        
    def shape_curve(slope, shift, push):
        out = [generalized_logistic_curve(x=y - CURRENT_YEAR,
                                          slope=slope,
                                          shift=shift,
                                          push=push,
                                          maximum=maximum,
                                          minimum=minimum) for y in year_cuts]
        return -np.mean([np.abs(out[i] - desired[i]) for i in range(len(out))])


    pbounds = {'slope': (0.01, 10),
               'shift': (0.01, 10),
               'push': (0.01, 10)}
    optimizer = BayesianOptimization(f=shape_curve, pbounds=pbounds, verbose=verbose, allow_duplicate_points=True)
    optimizer.maximize(init_points=40, n_iter=80)
    params = optimizer.max['params']
    if verbose:
        print('Curve params found')
        pprint(params)
        print('-')


    def p_nonscaling_delay(year):
        if year <= init_year:
            if init_year != CURRENT_YEAR:
                return 0
            else:
                return minimum
        elif year >= bottom_year:
            return maximum
        else:
            return generalized_logistic_curve(x=year - CURRENT_YEAR,
                                              slope=params['slope'],
                                              shift=params['shift'],
                                              push=params['push'],
                                              maximum=maximum,
                                              minimum=minimum)

    return p_nonscaling_delay


def plot_nonscaling_delay(plt, years, p_nonscaling_delay):
    print('## Chance of nonscaling delay ##')
    if callable(p_nonscaling_delay):
        p_delay_ = np.array([p_nonscaling_delay(y) for y in years])
    else:
        p_delay_ = np.array([p_nonscaling_delay for y in years])
    plt.plot(years, p_delay_, color='black')
    plt.ylabel('chance of a non-scaling delay')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - chance of a nonscaling delay if TAI compute needs are otherwise met in this year: {}%'
        print(outstr.format(y, int(round(p_delay_[y - CURRENT_YEAR] * 100))))


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
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
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


def calculate_nonscaling_delay(y, nonscaling_delay_, variables, print_diagnostic):
    is_nonscaling_issue = False
    nonscaling_delay_out = 0
    nonscaling_countdown = 0
    if nonscaling_delay_ is not None:
        if isinstance(nonscaling_delay_, dict):
            if len(nonscaling_delay_) == 1:
                nonscaling_delay_ = list(nonscaling_delay_.items())[0][1]
                p_nonscaling_delay_ = nonscaling_delay_['prob']
                # TODO: executing an arbitrary function from cache is not good
                if callable(p_nonscaling_delay_):
                    p_nonscaling_delay_ = np.array([p_nonscaling_delay_(y) for y in years])
                else:
                    p_nonscaling_delay_ = np.array([p_nonscaling_delay_ for y in years])
                p_nonscaling_delay_ = p_nonscaling_delay_[y - variables['CURRENT_YEAR']]
                is_nonscaling_issue = sq.event(p_nonscaling_delay_)
                nonscaling_delay_ = nonscaling_delay_['length']
                nonscaling_countdown = nonscaling_delay_
                if print_diagnostic:
                    print('-- {} p_nonscaling_issue={}'.format('Nonscaling delay occured' if is_nonscaling_issue else 'Nonscaling issue did not occur',
                                                               np.round(p_nonscaling_delay_, 4)))
            else:
                nonscaling_delay__ = nonscaling_delay_
                for name, delay in nonscaling_delay__.items():
                    p_nonscaling_delay_ = delay['prob']
                    if callable(p_nonscaling_delay_):
                        p_nonscaling_delay_ = np.array([p_nonscaling_delay_(y) for y in years])
                    else:
                        p_nonscaling_delay_ = np.array([p_nonscaling_delay_ for y in years])
                    # TODO: executing an arbitrary function from cache is not good
                    p_nonscaling_delay_ = p_nonscaling_delay_[y - variables['CURRENT_YEAR']]
                    this_nonscaling_issue = sq.event(p_nonscaling_delay_)
                    if print_diagnostic:
                        print('-- {} p_nonscaling_issue p={} -> {}'.format(name,
                                                                           np.round(p_nonscaling_delay_, 4),
                                                                           'Nonscaling delay occured' if this_nonscaling_issue else 'Nonscaling issue did not occur',))
                    if this_nonscaling_issue:
                        if not is_nonscaling_issue:
                            is_nonscaling_issue = True
                            nonscaling_delay_ = sq.sample(delay['length'])
                            if print_diagnostic:
                                print('-- -- this delay is {} years (total delay {} years)'.format(int(np.ceil(nonscaling_delay_)),
                                                                                                   int(np.ceil(nonscaling_delay_))))
                            nonscaling_delay_out = nonscaling_delay_
                            nonscaling_countdown = nonscaling_delay_
                        else:
                            this_nonscaling_delay = sq.sample(delay['length'])
                            nonscaling_delay_out = nonscaling_delay_
                            max_delay = np.max([nonscaling_delay_, this_nonscaling_delay])
                            nonscaling_delay_ = int(np.ceil(max_delay + this_nonscaling_delay / 4))
                            if print_diagnostic:
                                print('-- -- this delay is {} years (total delay {} years)'.format(int(np.ceil(this_nonscaling_delay)),
                                                                                                   int(np.ceil(nonscaling_delay_))))
                            nonscaling_delay_out = nonscaling_delay_
                            nonscaling_countdown = nonscaling_delay_
        else:
            raise ValueError('nonscaling delay information must be passed as a dictionary')

    return {'is_nonscaling_issue': is_nonscaling_issue,
            'nonscaling_delay_out': nonscaling_delay_out,
            'nonscaling_countdown': nonscaling_countdown}


def p_event(variables, label, verbosity):
    if isinstance(variables, dict):
        p = variables[label]
    else:
        p = variables
    outcome = sq.event(p)
    if verbosity > 1:
        print('-- sampling {} p={} outcome={}'.format(label, p, outcome))
    return outcome




def plot_model_versus_estimate(model_name, model_samples, actual_spend):
    print('## Model predicts {} spend will be ##'.format(model_name))
    pprint(dict([(i[0], numerize(10 ** i[1])) for i in sq.get_percentiles(model_samples).items()]))
    print('-')
    
    estimate_cost_samples = sq.dist_fn(sq.lognorm(actual_spend/4, actual_spend*4), fn=np.log10) @ (100*K)
    print('## Actual {} estimated to be ##'.format(model_name))
    pprint(dict([(i[0], numerize(10 ** i[1])) for i in sq.get_percentiles(estimate_cost_samples).items()]))
    print('-')
    
    print('Actual spend on {} (${}M) is at the {}th percentile of the model'.format(model_name,
                                                                                    round(actual_spend / M, 1),
                                                                                    round(np.mean([s <= np.log10(actual_spend) for s in model_samples]) * 100, 1)))
    print('-')
    
    plt.figure(figsize=(10,8))
    plt.hist(model_samples, bins=200, label='{} Prediction from Model'.format(model_name), alpha=0.6, color='blue')
    plt.axvline(np.mean(model_samples), label='{} Prediction from Model (mean)'.format(model_name), color='blue')
    plt.hist(estimate_cost_samples, bins=200, label='{} Estimates of cost'.format(model_name), alpha=0.6, color='orange')
    plt.axvline(np.mean(estimate_cost_samples), label='{} Prediction from Model (mean)'.format(model_name), color='orange')
    plt.xlabel('log $ spent')
    plt.legend()
    plt.show()
    return None


def show_model_forecast(samples):
    mean_ci = sq.get_mean_and_ci(samples, credibility=80)
    print('${} (80%CI: ${} to ${})'.format(numerize(10 ** mean_ci['mean']),
                                           numerize(10 ** mean_ci['ci_low']),
                                           numerize(10 ** mean_ci['ci_high'])))
    print('-')
    pprint(dict([(i[0], numerize(10 ** i[1])) for i in sq.get_percentiles(samples).items()]))
    print('-')
    plt.hist(samples, bins=200)
    plt.xlabel('log $ spent')
    plt.show()
    return None
