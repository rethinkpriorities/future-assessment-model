import math
import numpy as np
from bayes_opt import BayesianOptimization


def gdp(initial_gdp, gdp_growth, year):
    return initial_gdp * (gdp_growth ** year)


def willingness_to_pay(initial_gdp, gdp_growth, initial_pay, spend_doubling_time, max_gdp_frac, year):
    gdp_ = gdp(initial_gdp=initial_gdp, gdp_growth=gdp_growth, year=year)
    x = (np.log(2) / spend_doubling_time) * year
    if x < 650:
        y = math.log10(initial_pay) + math.log10(math.exp(x)) - math.log10(1 + initial_pay / (gdp_ * max_gdp_frac) * math.exp(x))
        if y > 300:
            y = int(y) # Handle overflow errors
        return 10 ** y
    else: # Handle math.exp and math.log10 overflow errors
        return 10 ** int(math.log10(initial_pay) + (year/spend_doubling_time)/3.3)
    

def algo_halving_fn(min_speed, max_speed, tai_flop_size):
    if max_speed < min_speed:
        max_speed = min_speed
    if min_speed > max_speed:
        min_speed = max_speed
    return min(max(max_speed - (round((tai_flop_size - 29) / 2) / 2), min_speed), max_speed)


def flop_needed(initial_flop, possible_reduction, doubling_rate, year):
    x = (np.log(2) / doubling_rate) * year
    if x < 650:
        y = (math.log10(initial_flop) - max(math.log10(math.exp(x)) - math.log10(1 + (1/possible_reduction) * math.exp(x)), 0))
        if y > 300:
            y = int(y) # Handle overflow errors        
        return 10 ** y
    else: # Handle math.exp and math.log10 overflow errors
        return 10 ** int(math.log10(initial_flop) - (1/possible_reduction))

    
def flop_per_dollar(initial_flop_per_dollar, max_flop_per_dollar, halving_rate, year):
    x = (np.log(2) / halving_rate) * year
    if x < 650:
        y = (math.log10(initial_flop_per_dollar) + math.log10(math.exp(x)) - math.log10(1 + initial_flop_per_dollar / max_flop_per_dollar * math.exp(x)))
        if y > 300:
            y = int(y) # Handle overflow errors                
        return 10 ** y
    else: # Handle math.exp and math.log10 overflow errors
        return 10 ** int(math.log10(initial_flop_per_dollar) + (year/halving_rate)/3.3)

    
def cost_of_tai(initial_flop, possible_reduction, algo_doubling_rate, initial_flop_per_dollar, max_flop_per_dollar,
                flop_halving_rate, year):
    return (flop_needed(initial_flop, possible_reduction, algo_doubling_rate, year) /
            flop_per_dollar(initial_flop_per_dollar, max_flop_per_dollar, flop_halving_rate, year))


def flop_at_max(initial_gdp, gdp_growth, initial_pay, spend_doubling_time, max_gdp_frac,
                 initial_flop_per_dollar, max_flop_per_dollar, flop_halving_rate, year):
    return (willingness_to_pay(initial_gdp=initial_gdp,
                               gdp_growth=gdp_growth,
                               initial_pay=initial_pay,
                               spend_doubling_time=spend_doubling_time,
                               max_gdp_frac=max_gdp_frac,
                               year=year) *
            flop_per_dollar(initial_flop_per_dollar, max_flop_per_dollar, flop_halving_rate, year))


def possible_algo_reduction_fn(min_reduction, max_reduction, tai_flop_size):
    if max_reduction < min_reduction:
        max_reduction = min_reduction
    if min_reduction > max_reduction:
        min_reduction = max_reduction
    return min(max(min_reduction + round((tai_flop_size - 32) / 4), min_reduction), max_reduction)


def derive_nonscaling_delay_curve(minimum, maximum, bottom_year, verbose=True):
    def shape_curve(slope, shift, push):
        years = list(range(CURRENT_YEAR, bottom_year))
        year_cuts = [years[0], years[int(round(len(years) / 2))], years[-1]]
        
        out = [generalized_logistic_curve(x=y - CURRENT_YEAR,
                                          slope=slope,
                                          shift=shift,
                                          push=push,
                                          maximum=maximum,
                                          minimum=minimum) for y in year_cuts]
        return -np.mean([np.abs(minimum - out[0]) ** 2,
                         np.abs(((minimum + maximum) / 2) - out[1]) ** 2,
                         np.abs(maximum - out[2]) ** 2])

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
        if year == CURRENT_YEAR:
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


def run_tai_model_round(initial_gdp_, tai_flop_size_, algo_doubling_rate_, possible_algo_reduction_,
                        initial_flop_per_dollar_, flop_halving_rate_, max_flop_per_dollar_, initial_pay_,
                        gdp_growth_, max_gdp_frac_, willingness_ramp_, spend_doubling_time_, p_nonscaling_delay,
                        nonscaling_delay_, willingness_spend_horizon_, print_diagnostic):
    queue_tai_year = 99999
    plt.ioff()
    if print_diagnostic:
        cost_of_tai_collector = []
        willingness_collector = []
    
    if print_diagnostic:
        print('It takes {} log FLOP (~{}) for transformative capabilities.'.format(np.round(tai_flop_size_, 1),
                                                                                    numerize(10 ** tai_flop_size_)))
        print('Every {} years algorithms get 2x better, with {} log reductions possible.'.format(np.round(algo_doubling_rate_, 1),
                                                                                                 np.round(possible_algo_reduction_, 1)))
        print(('FLOP start at a cost of {} log FLOP (~{}) per 2022$USD. Every {} years they get ' +
               '2x cheaper, to a maximum of {} log FLOP (~{}) per 2022$USD.').format(np.round(math.log10(initial_flop_per_dollar_), 1),
                                                                               numerize(initial_flop_per_dollar_),
                                                                               np.round(flop_halving_rate_, 1),
                                                                               np.round(math.log10(max_flop_per_dollar_), 1),
                                                                               numerize(max_flop_per_dollar_)))
        print(('We are willing to pay {} log 2022$USD (~{}) and this doubles every {} years to a max of {}% of GDP. ' +
               'GDP grows at a rate of {}x per year.').format(np.round(math.log10(initial_pay_), 1),
                                                              numerize(initial_pay_),
                                                              np.round(spend_doubling_time_, 1),
                                                              np.round(max_gdp_frac_, 6),
                                                              np.round(gdp_growth_, 3)))
        if willingness_ramp_ < 1:
            print('In this simulation, if we are {}% of the way to paying for TAI, we will ramp to paying for TAI.'.format(np.round(willingness_ramp_ * 100)))

        if willingness_spend_horizon_ > 1:
            print('We are willing to spend over {} years to make TAI'.format(willingness_spend_horizon_))
            
        print(('If a non-scaling delay happens, it will take an additional {} years to produce TAI due' +
               ' to issues unrelated to scaling FLOP').format(np.round(nonscaling_delay_, 1)))
        print('---')
    
    tai_created = False
    is_nonscaling_issue = None
    for y in years:
        if not tai_created:
            flop_needed_ = flop_needed(initial_flop=10 ** tai_flop_size_,
                                         doubling_rate=algo_doubling_rate_,
                                         possible_reduction=10 ** possible_algo_reduction_,
                                         year=(y - CURRENT_YEAR))
            
            flop_per_dollar_ = flop_per_dollar(initial_flop_per_dollar=initial_flop_per_dollar_,
                                                 max_flop_per_dollar=max_flop_per_dollar_,
                                                 halving_rate=flop_halving_rate_,
                                                 year=(y - CURRENT_YEAR))
            
            if flop_per_dollar_ > 10 ** 200 or flop_needed_ > 10 ** 200:
                flop_needed_ = int(flop_needed_)
                flop_per_dollar_ = int(flop_per_dollar_)
                cost_of_tai_ = flop_needed_ // flop_per_dollar_
            else:
                cost_of_tai_ = flop_needed_ / flop_per_dollar_
            
            willingness_ = willingness_to_pay(initial_gdp=initial_gdp_,
                                              gdp_growth=gdp_growth_,
                                              initial_pay=initial_pay_,
                                              spend_doubling_time=spend_doubling_time_,
                                              max_gdp_frac=max_gdp_frac_,
                                              year=(y - CURRENT_YEAR))
            
            if flop_per_dollar_ > 10 ** 200:
                willingness_ = int(willingness_)
            if willingness_ > 10 ** 200:
                flop_per_dollar_ = int(flop_per_dollar_)
            
            if print_diagnostic:
                cost_of_tai_collector.append(cost_of_tai_)
                willingness_collector.append(willingness_)
            
            total_compute_ = willingness_ * flop_per_dollar_
            
            if print_diagnostic:
                out_str = ('Year: {} - {} max log FLOP ({}) available - TAI takes {} log FLOP ({}) - ' +
                           'log $ {} to buy TAI ({}) vs. willingness to pay log $ {} ({}) - {} log FLOP per $ ({})')
                print(out_str.format(y,
                                     np.round(math.log10(total_compute_), 1),
                                     numerize(total_compute_),
                                     np.round(math.log10(flop_needed_), 1),
                                     numerize(flop_needed_),
                                     np.round(math.log10(cost_of_tai_), 1),
                                     numerize(cost_of_tai_),
                                     np.round(math.log10(willingness_), 1),
                                     numerize(willingness_),
                                     np.round(math.log10(flop_per_dollar_), 1),
                                     numerize(flop_per_dollar_)))
            
            if cost_of_tai_ > 10 ** 200:
                spend_tai_years = int(cost_of_tai_) // int(willingness_)
            else:
                spend_tai_years = cost_of_tai_ / willingness_
                
            if not is_nonscaling_issue and queue_tai_year < 99999 and print_diagnostic:
                print('-$- {}/{}'.format(y, queue_tai_year))
            if (cost_of_tai_ * willingness_ramp_) <= willingness_ or y >= queue_tai_year:
                if is_nonscaling_issue is None:
                    p_nonscaling_delay_ = p_nonscaling_delay(y) if p_nonscaling_delay is not None else 0
                    is_nonscaling_issue = sq.event(p_nonscaling_delay_)
                    nonscaling_countdown = nonscaling_delay_
                    if print_diagnostic:
                        print('-- {} p_nonscaling_issue={}'.format('Nonscaling delay occured' if is_nonscaling_issue else 'Nonscaling issue did not occur',
                                                                   np.round(p_nonscaling_delay_, 4)))
                
                if not is_nonscaling_issue or nonscaling_countdown <= 0.1:
                    if print_diagnostic:
                        print('--- /!\ TAI CREATED in {}'.format(y))
                        plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()
                    return y
                else:
                    if print_diagnostic:
                        print('/!\ FLOP for TAI sufficient but needs {} more years to solve non-scaling issues'.format(np.round(nonscaling_countdown, 1)))
                    nonscaling_countdown -= 1
            elif (not is_nonscaling_issue and willingness_spend_horizon_ > 1 and
                  spend_tai_years <= willingness_spend_horizon_ and y + math.ceil(spend_tai_years) < queue_tai_year):
                queue_tai_year = y + math.ceil(spend_tai_years)
                if print_diagnostic:
                    print('-$- We have enough spend to make TAI in {} years (in {}) if sustained.'.format(math.ceil(spend_tai_years),
                                                                                                          queue_tai_year))
                
    if not tai_created:
        if print_diagnostic:
            print('--- :/ TAI NOT CREATED BEFORE {}'.format(MAX_YEAR + 1))
            plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()
        return MAX_YEAR + 1


def print_graph(samples, label, reverse=False, digits=1):
    pctiles = sq.get_percentiles(samples, reverse=reverse, digits=digits)
    if len(set(samples)) > 1:
        print('## {} ##'.format(label))
        pprint(pctiles)
        plt.hist(samples, bins = 200)
        plt.show()
        print('-')
        print('-')
    else:
        print('## {}: {} ##'.format(label, samples[0]))
        print('-')
    return pctiles


def print_tai_arrival_stats(tai_years):
    print('## DISTRIBUTION OF TAI ARRIVAL DATE ##')
    pctiles = sq.get_percentiles(tai_years, percentiles=[5, 10, 15, 20, 25, 35, 50, 60, 75, 80, 90, 95])
    pprint([str(o[0]) + '%: ' + (str(int(o[1])) if o[1] < MAX_YEAR else '>' + str(MAX_YEAR)) for o in pctiles.items()])
    print('-')
    print('-')

    print('## DISTRIBUTION OF RELATIVE TAI ARRIVAL DATE ##')
    pprint([str(o[0]) + '%: ' + (str(int(o[1]) - CURRENT_YEAR) if o[1] < MAX_YEAR else '>' + str(MAX_YEAR - CURRENT_YEAR)) + ' years from now' for o in pctiles.items()])
    print('-')
    print('-')


    print('## TAI ARRIVAL DATE BY BIN ##')

    def bin_tai_yrs(low=None, hi=None):
        low = CURRENT_YEAR if low is None else low
        if hi is None:
            r = np.mean([y >= low for y in tai_years])
        else:
            r = np.mean([(y >= low) and (y <= hi) for y in tai_years])
        return round(r * 100, 1)


    print('This year: {}%'.format(bin_tai_yrs(hi=CURRENT_YEAR)))
    print('2024-2027: {}%'.format(bin_tai_yrs(2024, 2026)))
    print('2028-2029: {}%'.format(bin_tai_yrs(2027, 2029)))
    print('2030-2034: {}%'.format(bin_tai_yrs(2030, 2034)))
    print('2035-2039: {}%'.format(bin_tai_yrs(2035, 2039)))
    print('2040-2049: {}%'.format(bin_tai_yrs(2040, 2049)))
    print('2050-2059: {}%'.format(bin_tai_yrs(2050, 2059)))
    print('2060-2069: {}%'.format(bin_tai_yrs(2060, 2069)))
    print('2070-2079: {}%'.format(bin_tai_yrs(2070, 2079)))
    print('2080-2089: {}%'.format(bin_tai_yrs(2080, 2089)))
    print('2090-2099: {}%'.format(bin_tai_yrs(2090, 2099)))
    print('2100-2109: {}%'.format(bin_tai_yrs(2100, 2109)))
    print('2110-2119: {}%'.format(bin_tai_yrs(2110, 2119)))
    print('>2120: {}%'.format(bin_tai_yrs(low=2120)))
    print('-')

    print('## TAI ARRIVAL DATE BY YEAR - COMPARE TO BENCHMARK ##')
    print('By EOY 2024: {}%'.format(bin_tai_yrs(hi=2024)))
    print('By EOY 2025: {}%'.format(bin_tai_yrs(hi=2025)))
    print('By EOY 2027: {}% (within 5 yrs)'.format(bin_tai_yrs(hi=2027)))
    print('By EOY 2030: {}% (Ajeya 2022: 15%)'.format(bin_tai_yrs(hi=2030)))
    print('By EOY 2032: {}% (within 10yrs)'.format(bin_tai_yrs(hi=2032)))
    print('By EOY 2036: {}% (Holden 2021 benchmark - 10%-50%, Holden 2021: 10%; Ajeya 2022: 35%)'.format(bin_tai_yrs(hi=2036)))
    print('By EOY 2040: {}% (Ajeya 2022: 50%)'.format(bin_tai_yrs(hi=2040)))
    print('By EOY 2042: {}% (FTX: 20%, 10%-45%)'.format(bin_tai_yrs(hi=2042)))
    print('By EOY 2047: {}% (within 25yrs)'.format(bin_tai_yrs(hi=2047)))
    print('By EOY 2050: {}% (Ajeya 2020: 50%, Ajeya 2022: 60%)'.format(bin_tai_yrs(hi=2050)))
    print('By EOY 2060: {}% (Holden 2021 benchmark - 25%-75%, Holden 2021: 50%)'.format(bin_tai_yrs(hi=2060)))
    print('By EOY 2070: {}% (Carlsmith: 50%)'.format(bin_tai_yrs(hi=2070)))
    print('By EOY 2072: {}% (within 50yrs)'.format(bin_tai_yrs(hi=2072)))
    print('By EOY 2078: {}% (within my expected lifetime)'.format(bin_tai_yrs(hi=2078)))
    print('By EOY 2099: {}% (FTX: 60%, >30%)'.format(bin_tai_yrs(hi=2099)))
    print('By EOY 2100: {}% (Holden 2021 benchmark - 33%-90%, Holden 2021: 66%)'.format(bin_tai_yrs(hi=2100)))
    print('By EOY 2122: {}% (within 100yrs)'.format(bin_tai_yrs(hi=2122)))
    print('-')
    print('-')

    tai_years_ = np.array([MAX_YEAR + 1 if t > MAX_YEAR else t for t in tai_years])
    count, bins_count = np.histogram(tai_years_, bins=(MAX_YEAR - CURRENT_YEAR))
    bins = np.round(np.array([b for b in bins_count[1:] if b <= MAX_YEAR]))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    plt.plot(bins, cdf[:len(bins)], label='CDF')
    plt.legend()
    plt.show()

    pdf_smoothed = savitzky_golay(pdf[:len(bins)], 51, 3)
    plt.plot(bins, pdf_smoothed, label='PDF (smoothed)')
    plt.legend()
    plt.show()


def run_timelines_model(variables, cores=1, runs=10000, load_cache_file=None,
                        dump_cache_file=None, reload_cache=False):

    initial_chance_of_nonscaling_issue_ = variables.get('initial_chance_of_nonscaling_issue', 0)
    final_chance_of_nonscaling_issue_ = variables.get('final_chance_of_nonscaling_issue', 0)
    nonscaling_issue_bottom_year_ = variables.get('nonscaling_issue_bottom_year', 0)
    if nonscaling_issue_bottom_year_ == 0:
        p_nonscaling_delay = None
    else:
        print('Deriving nonscaling delay curve...')
        p_nonscaling_delay = derive_nonscaling_delay_curve(minimum=initial_chance_of_nonscaling_issue_,
                                                           maximum=final_chance_of_nonscaling_issue_,
                                                           bottom_year=nonscaling_issue_bottom_year_)

    def define_event(verbose=False):
        tai_flop_size_ = variables['tai_flop_size']
        if isinstance(tai_flop_size_, sq.BaseDistribution):
            tai_flop_size_ = sq.sample(tai_flop_size_)
        else:
            tai_flop_size_ = sq.sample(sq.discrete(variables['tai_flop_size']))

        if tai_flop_size_ > 300:
            tai_flop_size_ = int(tai_flop_size_) # Handle overflow errors
        
        algo_doubling_rate_ = algo_halving_fn(sq.sample(variables['algo_doubling_rate_min']),
                                              sq.sample(variables['algo_doubling_rate_max']),
                                              tai_flop_size_)
        
        possible_algo_reduction_ = possible_algo_reduction_fn(sq.sample(variables['min_reduction']),
                                                              sq.sample(variables['max_reduction']),
                                                              tai_flop_size_)
        
        initial_flop_per_dollar_ = 10 ** sq.sample(variables['initial_flop_per_dollar'])
        flop_halving_rate_ = sq.sample(variables['flop_halving_rate'])
        max_flop_per_dollar_ = 10 ** sq.sample(variables['max_flop_per_dollar'])
        initial_pay_ = 10 ** sq.sample(variables['initial_pay'])
        gdp_growth_ = sq.sample(variables['gdp_growth'])
        max_gdp_frac_ = sq.sample(variables['max_gdp_frac'])
        
        willingness_ramp_happens = sq.event_occurs(variables.get('p_willingness_ramp', 0))
        if willingness_ramp_happens:
            willingness_ramp_ = sq.sample(variables.get('willingness_ramp', 1))
        else:
            willingness_ramp_ = 1
        
        initial_gdp_ = variables['initial_gdp']
        spend_doubling_time_ = sq.sample(variables['spend_doubling_time'])
        nonscaling_delay_ = sq.sample(variables.get('nonscaling_delay', 0))
        willingness_spend_horizon_ = int(sq.sample(variables.get('willingness_spend_horizon', 1)))
        
        return run_tai_model_round(initial_gdp_=initial_gdp_,
                                   tai_flop_size_=tai_flop_size_,
                                   algo_doubling_rate_=algo_doubling_rate_,
                                   possible_algo_reduction_=possible_algo_reduction_,
                                   initial_flop_per_dollar_=initial_flop_per_dollar_,
                                   flop_halving_rate_=flop_halving_rate_,
                                   max_flop_per_dollar_=max_flop_per_dollar_,
                                   initial_pay_=initial_pay_,
                                   gdp_growth_=gdp_growth_,
                                   max_gdp_frac_=max_gdp_frac_,
                                   willingness_ramp_=willingness_ramp_,
                                   spend_doubling_time_=spend_doubling_time_,
                                   p_nonscaling_delay=p_nonscaling_delay,
                                   nonscaling_delay_=nonscaling_delay_,
                                   willingness_spend_horizon_=willingness_spend_horizon_,
                                   print_diagnostic=verbose)

    print('## RUN TIMELINES MODEL ##')
    tai_years = bayes.bayesnet(define_event,
                               verbose=True,
                               raw=True,
                               cores=cores,
                               load_cache_file=load_cache_file,
                               dump_cache_file=dump_cache_file,
                               reload_cache=reload_cache,
                               n=runs)

    print('-')
    print_tai_arrival_stats(tai_years)
    print('-')
    print('-')

    initial_flop_s = variables['tai_flop_size']
    if isinstance(initial_flop_s, sq.BaseDistribution):
        initial_flop_s = sq.sample(initial_flop_s, n=1000)
    else:
        initial_flop_s = sq.sample(sq.discrete(initial_flop_s), n=1000)
    initial_flop_p = print_graph(initial_flop_s, 'TAI FLOP SIZE')

    min_reduction_s = sq.sample(variables['min_reduction'], n=1000)
    min_reduction_p = print_graph(min_reduction_s, 'MIN REDUCTION')

    max_reduction_s = sq.sample(variables['max_reduction'], n=1000)
    max_reduction_p = print_graph(max_reduction_s, 'MAX REDUCTION', reverse=True)

    algo_doubling_rate_min_s = sq.sample(variables['algo_doubling_rate_min'], n=1000)
    algo_doubling_rate_min_p = print_graph(algo_doubling_rate_min_s,
                                           label='MIN ALGO DOUBLING RATE',
                                           reverse=True)

    algo_doubling_rate_max_s = sq.sample(variables['algo_doubling_rate_max'], n=1000)
    algo_doubling_rate_max_p = print_graph(algo_doubling_rate_max_s,
                                           label='MAX ALGO DOUBLING RATE',
                                           reverse=True)

    initial_flop_per_dollar_s = sq.sample(variables['initial_flop_per_dollar'], n=1000)
    initial_flop_per_dollar_p = print_graph(initial_flop_per_dollar_s,
                                             label='INITIAL FLOP PER DOLLAR')

    flop_halving_rate_s = sq.sample(variables['flop_halving_rate'], n=1000)
    flop_halving_rate_p = print_graph(flop_halving_rate_s,
                                       label='FLOP HALVING RATE',
                                       reverse=True)

    max_flop_per_dollar_s = sq.sample(variables['max_flop_per_dollar'], n=1000)
    max_flop_per_dollar_p = print_graph(max_flop_per_dollar_s, label='MAX FLOP PER DOLLAR')

    initial_pay_s = sq.sample(variables['initial_pay'], n=1000)
    initial_pay_p = print_graph(initial_pay_s, label='INITIAL PAY')

    gdp_growth_s = sq.sample(variables['gdp_growth'], n=1000)
    gdp_growth_p = print_graph(gdp_growth_s, label='GDP GROWTH', digits=2)

    max_gdp_frac_s = sq.sample(variables['max_gdp_frac'], n=1000)
    max_gdp_frac_p = print_graph(max_gdp_frac_s, label='MAX GDP FRAC', digits=5)

    if variables.get('nonscaling_delay', 0) != 0:
        nonscaling_delay_s = sq.sample(variables['nonscaling_delay'], n=1000)
        nonscaling_delay_p = print_graph(nonscaling_delay_s, label='NONSCALING DELAY', digits=0)

    if variables.get('initial_chance_of_nonscaling_issue', 0) != 0:
        initial_chance_of_nonscaling_issue_s = sq.sample(variables['initial_chance_of_nonscaling_issue'], n=1000)
        initial_chance_of_nonscaling_issue_p = print_graph(initial_chance_of_nonscaling_issue_s,
                                                           label='INITIAL CHANCE OF NONSCALING ISSUE',
                                                           digits=0)

    if variables.get('initial_chance_of_nonscaling_issue', 0) != 0:
        final_chance_of_nonscaling_issue_s = sq.sample(variables['final_chance_of_nonscaling_issue'], n=1000)
        final_chance_of_nonscaling_issue_p = print_graph(final_chance_of_nonscaling_issue_s,
                                                         label='FINAL CHANCE OF NONSCALING ISSUE',
                                                         digits=0)

    if variables.get('nonscaling_issue_bottom_year', 0) != 0:
        nonscaling_issue_bottom_year_s = sq.sample(variables['nonscaling_issue_bottom_year'], n=1000)
        nonscaling_issue_bottom_year_p = print_graph(nonscaling_issue_bottom_year_s,
                                                     label='NONSCALING BOTTOM YEAR',
                                                     digits=0)

    willingness_ramp = variables.get('willingness_ramp', 0)
    if willingness_ramp != 0:
        willingness_ramp_s = sq.sample(willingness_ramp, n=1000)
        willingness_ramp_p = print_graph(willingness_ramp_s, label='WILLINGNESS RAMP')

    spend_doubling_time_s = sq.sample(variables['spend_doubling_time'], n=1000)
    spend_doubling_time_p = print_graph(spend_doubling_time_s,
                                        label='SPEND DOUBLING TIME',
                                        reverse=True)

    willingness_spend_horizon = variables.get('willingness_spend_horizon', 1)
    if willingness_spend_horizon != 1:
        willingness_spend_horizon_s = sq.sample(willingness_spend_horizon, n=1000)
        willingness_spend_horizon_p = print_graph(willingness_spend_horizon_s,
                                                  label='WILLINGNESS SPEND HORIZON')


    print('-')
    print('-')
    print('## GDP Over Time ##')
    gdp_50 = np.array([gdp(initial_gdp=variables['initial_gdp'],
                           gdp_growth=gdp_growth_p[50],
                           year=(y - CURRENT_YEAR)) for y in years])
    gdp_10 = np.array([gdp(initial_gdp=variables['initial_gdp'],
                           gdp_growth=gdp_growth_p[10],
                           year=(y - CURRENT_YEAR)) for y in years])
    gdp_90 = np.array([gdp(initial_gdp=variables['initial_gdp'],
                           gdp_growth=gdp_growth_p[90],
                           year=(y - CURRENT_YEAR)) for y in years])
    plt.plot(years, np.log10(gdp_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(gdp_50), color='black')
    plt.plot(years, np.log10(gdp_90), linestyle='dashed', color='black')
    plt.ylabel('log GDP')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - GDP log 2022$USD {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(gdp_50[y - CURRENT_YEAR]), 1),
                            numerize(gdp_50[y - CURRENT_YEAR]),
                            np.round(np.log10(gdp_10[y - CURRENT_YEAR]), 1),
                            numerize(gdp_10[y - CURRENT_YEAR]),
                            np.round(np.log10(gdp_90[y - CURRENT_YEAR]), 1),
                            numerize(gdp_90[y - CURRENT_YEAR])))

    print('-')
    print('-')
    print('## Willingness to Pay Over Time ##')
    willingness_50 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[50],
                                                  initial_pay=10 ** initial_pay_p[50],
                                                  spend_doubling_time=spend_doubling_time_p[50],
                                                  max_gdp_frac=max_gdp_frac_p[50],
                                                  year=(y - CURRENT_YEAR)) for y in years])
    willingness_10 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[10],
                                                  initial_pay=10 ** initial_pay_p[10],
                                                  spend_doubling_time=spend_doubling_time_p[10],
                                                  max_gdp_frac=max_gdp_frac_p[10],
                                                  year=(y - CURRENT_YEAR)) for y in years])
    willingness_90 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[90],
                                                  initial_pay=10 ** initial_pay_p[90],
                                                  spend_doubling_time=spend_doubling_time_p[90],
                                                  max_gdp_frac=max_gdp_frac_p[90],
                                                  year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(willingness_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(willingness_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(willingness_50), color='black')
    plt.ylabel('log 2022$USD/yr willing to spend on TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - willingness log 2022$USD per year {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(willingness_50[y - CURRENT_YEAR]), 1),
                            numerize(willingness_50[y - CURRENT_YEAR]),
                            np.round(np.log10(willingness_10[y - CURRENT_YEAR]), 1),
                            numerize(willingness_10[y - CURRENT_YEAR]),
                            np.round(np.log10(willingness_90[y - CURRENT_YEAR]), 1),
                            numerize(willingness_90[y - CURRENT_YEAR])))


    print('-')
    print('-')
    print('## FLOP Needed to Make TAI (Given Algorithmic Progress) ##')
    flops_50 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[50],
                                      doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                    algo_doubling_rate_max_p[50],
                                                                    initial_flop_p[50]),
                                      possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[50],
                                                                                          max_reduction_p[50],
                                                                                          initial_flop_p[50]),
                                      year=(y - CURRENT_YEAR)) for y in years])
    flops_10 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[10],
                                      doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                    algo_doubling_rate_max_p[10],
                                                                    initial_flop_p[10]),
                                      possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[10],
                                                                                          max_reduction_p[10],
                                                                                          initial_flop_p[10]),
                                      year=(y - CURRENT_YEAR)) for y in years])
    flops_90 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[90],
                                      doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                    algo_doubling_rate_max_p[90],
                                                                    initial_flop_p[90]),
                                      possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[90],
                                                                                          max_reduction_p[90],
                                                                                          initial_flop_p[90]),
                                      year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(flops_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flops_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flops_50), color='black')
    plt.ylabel('log FLOP needed to make TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - log FLOP needed for TAI {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flops_50[y - CURRENT_YEAR]), 1),
                            numerize(flops_50[y - CURRENT_YEAR]),
                            np.round(np.log10(flops_10[y - CURRENT_YEAR]), 1),
                            numerize(flops_10[y - CURRENT_YEAR]),
                            np.round(np.log10(flops_90[y - CURRENT_YEAR]), 1),
                            numerize(flops_90[y - CURRENT_YEAR])))

    print('-')
    print('-')
    print('## FLOP per Dollar (Given Declining Costs) ##')
    flop_per_dollar_50 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                                     max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                                     halving_rate=flop_halving_rate_p[50],
                                                     year=(y - CURRENT_YEAR)) for y in years])
    flop_per_dollar_90 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                                     max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                                     halving_rate=flop_halving_rate_p[90],
                                                     year=(y - CURRENT_YEAR)) for y in years])
    flop_per_dollar_10 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                                     max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                                     halving_rate=flop_halving_rate_p[10],
                                                     year=(y - CURRENT_YEAR)) for y in years])
    plt.plot(years, np.log10(flop_per_dollar_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_per_dollar_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_per_dollar_50), color='black')
    plt.ylabel('log FLOP per $1')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - log FLOP per 2022$1USD {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flop_per_dollar_50[y - CURRENT_YEAR]), 1),
                            numerize(flop_per_dollar_50[y - CURRENT_YEAR]),
                            np.round(np.log10(flop_per_dollar_10[y - CURRENT_YEAR]), 1),
                            numerize(flop_per_dollar_10[y - CURRENT_YEAR]),
                            np.round(np.log10(flop_per_dollar_90[y - CURRENT_YEAR]), 1),
                            numerize(flop_per_dollar_90[y - CURRENT_YEAR])))

    print('-')
    print('-')
    print('## Max Possible OOM Reduction in TAI FLOP Size ##')
    tai_sizes = range(20, 50)
    flop_per_dollar_ = np.array([possible_algo_reduction_fn(min_reduction_p[50],
                                                             max_reduction_p[50], t) for t in tai_sizes])
    plt.plot(tai_sizes, flop_per_dollar_)
    plt.ylabel('max OOM reduction')
    plt.xlabel('initial FLOP needed for TAI prior to any reduction')
    plt.show()

    for t in tai_sizes:
        print('TAI log FLOP {} -> {} OOM reductions possible'.format(t,
                                                                     round(possible_algo_reduction_fn(min_reduction_p[50],
                                                                                                      max_reduction_p[50],
                                                                                                      t), 2)))

    print('-')
    print('-')
    print('## Halving time (years) of compute requirements ##')
    flop_per_dollar_ = np.array([algo_halving_fn(algo_doubling_rate_min_p[50],
                                                  algo_doubling_rate_max_p[50],
                                                  t) for t in tai_sizes])
    plt.plot(tai_sizes, flop_per_dollar_)
    plt.ylabel('number of years for compute requirements to halve')
    plt.show()

    for t in tai_sizes:
        print('TAI log FLOP {} -> algo doubling rate {}yrs'.format(t,
                                                                   round(algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                                         algo_doubling_rate_max_p[50],
                                                                                         t), 2)))

    if variables.get('initial_chance_of_nonscaling_issue', 0) != 0:
        print('-')
        print('-')
        print('## Chance of nonscaling delay ##')
        p_delay_ = np.array([p_nonscaling_delay(y) for y in years])
        plt.plot(years, p_delay_, color='black')
        plt.ylabel('chance of a non-scaling delay')
        plt.show()

        for y in years[:10] + years[10::10]:
            outstr = 'Year: {} - chance of a nonscaling delay if TAI compute needs are otherwise met in this year: {}%'
            print(outstr.format(y, int(round(p_delay_[y - CURRENT_YEAR] * 100))))

    print('-')
    print('-')
    print('## Dollars Needed to Buy TAI (Given Algorithmic Progress and Decline in Cost per FLOP) ##')
    cost_of_tai_50 = np.array([cost_of_tai(initial_flop=10 ** initial_flop_p[50],
                                           possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[50], max_reduction_p[50], initial_flop_p[50]),
                                           algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                              algo_doubling_rate_max_p[50],
                                                                              initial_flop_p[50]),
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                           flop_halving_rate=flop_halving_rate_p[50],
                                           year=(y - CURRENT_YEAR)) for y in years])
    cost_of_tai_10 = np.array([cost_of_tai(initial_flop=10 ** initial_flop_p[10],
                                           possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[10], max_reduction_p[10], initial_flop_p[10]),
                                           algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                              algo_doubling_rate_max_p[10],
                                                                              initial_flop_p[10]),
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                           flop_halving_rate=flop_halving_rate_p[10],
                                           year=(y - CURRENT_YEAR)) for y in years])
    cost_of_tai_90 = np.array([cost_of_tai(initial_flop=10 ** initial_flop_p[90],
                                           possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[90], max_reduction_p[90], initial_flop_p[90]),
                                           algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                              algo_doubling_rate_max_p[90],
                                                                              initial_flop_p[90]),
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                           flop_halving_rate=flop_halving_rate_p[90],
                                           year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(cost_of_tai_50), color='black')
    plt.plot(years, np.log10(cost_of_tai_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(cost_of_tai_90), linestyle='dashed', color='black')
    plt.ylabel('log $ needed to buy TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - {} log 2022$USD to buy TAI (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(cost_of_tai_50[y - CURRENT_YEAR]), 1),
                            numerize(cost_of_tai_50[y - CURRENT_YEAR]),
                            np.round(np.log10(cost_of_tai_10[y - CURRENT_YEAR]), 1),
                            numerize(cost_of_tai_10[y - CURRENT_YEAR]),
                            np.round(np.log10(cost_of_tai_90[y - CURRENT_YEAR]), 1),
                            numerize(cost_of_tai_90[y - CURRENT_YEAR])))

    print('-')
    print('-')
    print('## FLOP at Max Spend ##')
    flop_at_max_50 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                             gdp_growth=gdp_growth_p[50],
                                             initial_pay=10 ** initial_pay_p[50],
                                             spend_doubling_time=spend_doubling_time_p[50],
                                             max_gdp_frac=max_gdp_frac_p[50],
                                             initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                             max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                             flop_halving_rate=flop_halving_rate_p[50],
                                             year=(y - CURRENT_YEAR)) for y in years])
    flop_at_max_10 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                             gdp_growth=gdp_growth_p[10],
                                             initial_pay=10 ** initial_pay_p[10],
                                             spend_doubling_time=spend_doubling_time_p[10],
                                             max_gdp_frac=max_gdp_frac_p[10],
                                             initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                             max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                             flop_halving_rate=flop_halving_rate_p[10],
                                             year=(y - CURRENT_YEAR)) for y in years])
    flop_at_max_90 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                             gdp_growth=gdp_growth_p[90],
                                             initial_pay=10 ** initial_pay_p[90],
                                             spend_doubling_time=spend_doubling_time_p[90],
                                             max_gdp_frac=max_gdp_frac_p[90],
                                             initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                             max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                             flop_halving_rate=flop_halving_rate_p[90],
                                             year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(flop_at_max_50), color='black')
    plt.plot(years, np.log10(flop_at_max_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_at_max_10), linestyle='dashed', color='black')
    plt.show()
    plt.ylabel('max log FLOP bought given willingness to spend')

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - max log FLOP {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flop_at_max_50[y - CURRENT_YEAR]), 1),
                            numerize(flop_at_max_50[y - CURRENT_YEAR]),
                            np.round(np.log10(flop_at_max_10[y - CURRENT_YEAR]), 1),
                            numerize(flop_at_max_10[y - CURRENT_YEAR]),
                            np.round(np.log10(flop_at_max_90[y - CURRENT_YEAR]), 1),
                            numerize(flop_at_max_90[y - CURRENT_YEAR])))

    for i in range(3):
        print('-')
        print('-')
        print('## SAMPLE RUN {} ##'.format(i + 1))
        define_event(verbose=True)

    return None
