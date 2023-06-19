import math
import numpy as np
from bayes_opt import BayesianOptimization


def plot_tai(plt, years, cost_of_tai_collector, willingness_collector):
    cost = np.log10(np.array(cost_of_tai_collector))
    willingness = np.log10(np.array(willingness_collector))
    plt.plot(years[:len(cost)], cost, label='Cost of TAI')
    plt.plot(years[:len(willingness)], willingness, label='Willingness to pay for TAI')
    plt.legend()
    plt.ylabel('log $')
    return plt


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


def effective_flop_at_max(initial_gdp, gdp_growth, initial_pay, spend_doubling_time, max_gdp_frac,
                          initial_flop_per_dollar, max_flop_per_dollar, flop_halving_rate,
                          initial_flop, possible_reduction, doubling_rate, year):
    wtp = willingness_to_pay(initial_gdp=initial_gdp,
                             gdp_growth=gdp_growth,
                             initial_pay=initial_pay,
                             spend_doubling_time=spend_doubling_time,
                             max_gdp_frac=max_gdp_frac,
                             year=year)
    fpd = flop_per_dollar(initial_flop_per_dollar, max_flop_per_dollar, flop_halving_rate, year)
    fn = (flop_needed(initial_flop, possible_reduction, doubling_rate, 0) /
          flop_needed(initial_flop, possible_reduction, doubling_rate, year))
    return wtp * fpd * fn


def possible_algo_reduction_fn(min_reduction, max_reduction, tai_flop_size):
    if max_reduction < min_reduction:
        max_reduction = min_reduction
    if min_reduction > max_reduction:
        min_reduction = max_reduction
    return min(max(min_reduction + round((tai_flop_size - 32) / 4), min_reduction), max_reduction)


# TODO: Refactor
def run_tai_model_round(initial_gdp_, tai_flop_size_, algo_doubling_rate_, possible_algo_reduction_,
                        initial_flop_per_dollar_, flop_halving_rate_, max_flop_per_dollar_, initial_pay_,
                        gdp_growth_, max_gdp_frac_, willingness_ramp_, spend_doubling_time_, spend_doubling_time_2025_,
                        nonscaling_delay_, willingness_spend_horizon_, print_diagnostic, variables):
    if print_diagnostic:
        cost_of_tai_collector = []
        willingness_collector = []

    queue_tai_year = 99999
    effective_flop_collector = []
    tai_created = False
    is_nonscaling_issue = None
    nonscaling_delay_out = 0
    initial_cost_of_tai_ = None
    plt.ioff()
    
    if not tai_created and print_diagnostic:
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
        # Ensure 2023-2025 era doubling time is always faster
        if spend_doubling_time_ > spend_doubling_time_2025_:
            spend_doubling_time_2025_ = spend_doubling_time_

        # Cap initial pay at max GDP frac
        initial_pay_ = willingness_to_pay(initial_gdp=initial_gdp_,
                                          gdp_growth=gdp_growth_,
                                          initial_pay=initial_pay_,
                                          spend_doubling_time=1,
                                          max_gdp_frac=max_gdp_frac_,
                                          year=0)

        if spend_doubling_time_2025_ != spend_doubling_time_:
            print(('We are initially willing to pay {} log 2022$USD (~{}). This doubles every {} years until 2025, and then doubles every {} years to a max of {}% of GDP. ' +
                   'GDP grows at a rate of {}x per year.').format(np.round(math.log10(initial_pay_), 1),
                                                                  numerize(initial_pay_),
                                                                  np.round(spend_doubling_time_2025_, 1),
                                                                  np.round(spend_doubling_time_, 1),
                                                                  np.round(max_gdp_frac_ * 100, 6),
                                                                  np.round(gdp_growth_, 3)))
        else:
            print(('We are initially willing to pay {} log 2022$USD (~{}). This doubles every {} years to a max of {}% of GDP. ' +
                   'GDP grows at a rate of {}x per year.').format(np.round(math.log10(initial_pay_), 1),
                                                                  numerize(initial_pay_),
                                                                  np.round(spend_doubling_time_, 1),
                                                                  np.round(max_gdp_frac_ * 100, 6),
                                                                  np.round(gdp_growth_, 3)))

        if willingness_ramp_ < 1:
            print('In this simulation, if we are {}% of the way to paying for TAI, we will ramp to paying for TAI.'.format(np.round(willingness_ramp_ * 100)))

        if willingness_spend_horizon_ > 1:
            print('We are willing to spend over {} years to make TAI'.format(willingness_spend_horizon_))
        print('---')
    
    effective_flop_ = 0
    for y in years:
        flop_needed_ = flop_needed(initial_flop=10 ** tai_flop_size_,
                                   doubling_rate=algo_doubling_rate_,
                                   possible_reduction=10 ** possible_algo_reduction_,
                                   year=(y - variables['CURRENT_YEAR']))
        
        flop_per_dollar_ = flop_per_dollar(initial_flop_per_dollar=initial_flop_per_dollar_,
                                           max_flop_per_dollar=max_flop_per_dollar_,
                                           halving_rate=flop_halving_rate_,
                                           year=(y - variables['CURRENT_YEAR']))

        overflow = flop_per_dollar_ > 10 ** 200 or flop_needed_ > 10 ** 200
        flop_needed_ = int(flop_needed_) if overflow else flop_needed_
        flop_per_dollar_ = int(flop_per_dollar_) if overflow else flop_per_dollar_
        cost_of_tai_ = flop_needed_ // flop_per_dollar_ if overflow else flop_needed_ / flop_per_dollar_

        if cost_of_tai_ <= 1:
            cost_of_tai_ = 1
        if initial_cost_of_tai_ is None:
            initial_cost_of_tai_ = cost_of_tai_

        cost_ratio_ = initial_cost_of_tai_ // cost_of_tai_ if overflow else initial_cost_of_tai_ // cost_of_tai_

        if variables['CURRENT_YEAR'] >= 2025:
            raise ValueError('CURRENT_YEAR >= 2025 not currently supported')

        if y <= 2025 or spend_doubling_time_2025_ == spend_doubling_time_:
            willingness_ = willingness_to_pay(initial_gdp=initial_gdp_,
                                              gdp_growth=gdp_growth_,
                                              initial_pay=initial_pay_,
                                              spend_doubling_time=spend_doubling_time_2025_,
                                              max_gdp_frac=max_gdp_frac_,
                                              year=y - variables['CURRENT_YEAR'])
        else:
            willingness_2025_ = willingness_to_pay(initial_gdp=initial_gdp_,
                                                   gdp_growth=gdp_growth_,
                                                   initial_pay=initial_pay_,
                                                   spend_doubling_time=spend_doubling_time_2025_,
                                                   max_gdp_frac=max_gdp_frac_,
                                                   year=2025 - variables['CURRENT_YEAR'])
            gdp_2025_ = 10 ** gdp(initial_gdp=np.log10(initial_gdp_), gdp_growth=gdp_growth_, year=2025 - variables['CURRENT_YEAR'])
            willingness_ = willingness_to_pay(initial_gdp=gdp_2025_,
                                              gdp_growth=gdp_growth_,
                                              initial_pay=willingness_2025_,
                                              spend_doubling_time=spend_doubling_time_,
                                              max_gdp_frac=max_gdp_frac_,
                                              year=y - 2025)
        
        if not tai_created and print_diagnostic:
            cost_of_tai_collector.append(cost_of_tai_)
            willingness_collector.append(willingness_)

        if willingness_ > 10 ** 150:
            willingness_ = int(10 ** 150)
            flop_per_dollar_ = int(flop_per_dollar_)
            cost_ratio_ = int(cost_ratio_)

        if flop_per_dollar_ > 10 ** 150:
            flop_per_dollar_ = int(10 ** 150)
            willingness_ = int(willingness_)
            cost_ratio_ = int(cost_ratio_)

        total_compute_ = willingness_ * flop_per_dollar_

        effective_flop_ = willingness_ * initial_flop_per_dollar_ * cost_ratio_
        effective_flop_collector.append(effective_flop_)
        
        if not tai_created and print_diagnostic:
            out_str = ('Year: {} - {} max log FLOP ({}) available - TAI takes {} log FLOP ({}) - ' +
                       'log $ {} to buy TAI ({}) vs. willingness to pay log $ {} ({}) - {} log FLOP per $ ({}) (Effective 2023-logFLOP: {})')
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
                                 numerize(flop_per_dollar_),
                                 np.round(math.log10(effective_flop_), 1)))
        
        if cost_of_tai_ > 10 ** 200:
            spend_tai_years = int(cost_of_tai_) // int(willingness_)
        else:
            spend_tai_years = cost_of_tai_ / willingness_
            
        if not tai_created and not is_nonscaling_issue and queue_tai_year < 99999 and print_diagnostic:
            print('-$- {}/{}'.format(y, queue_tai_year))
        if (not tai_created and
            ((cost_of_tai_ * willingness_ramp_) <= willingness_ or y >= queue_tai_year)):
            if is_nonscaling_issue is None:
                delay_data = calculate_nonscaling_delay(y, nonscaling_delay_, variables, print_diagnostic)
                is_nonscaling_issue = delay_data['is_nonscaling_issue']
                nonscaling_delay_out = delay_data['nonscaling_delay_out']
                nonscaling_countdown = delay_data['nonscaling_countdown']
                
            if not is_nonscaling_issue or nonscaling_countdown <= 0.1:
                if print_diagnostic:
                    print('--- /!\ TAI CREATED in {}'.format(y))
                    plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()
                tai_created = True
                tai_year = y
            else:
                if print_diagnostic:
                    print('/!\ FLOP for TAI sufficient but needs {} more years to solve non-scaling issues'.format(int(np.ceil(nonscaling_countdown))))
                nonscaling_countdown -= 1
        elif (not is_nonscaling_issue and willingness_spend_horizon_ > 1 and
              spend_tai_years <= willingness_spend_horizon_ and y + math.ceil(spend_tai_years) < queue_tai_year):
            queue_tai_year = y + math.ceil(spend_tai_years)
            if print_diagnostic:
                print('-$- We have enough spend to make TAI in {} years (in {}) if sustained.'.format(math.ceil(spend_tai_years),
                                                                                                          queue_tai_year))
                
    if not tai_created:
        tai_year = variables['MAX_YEAR'] + 1
        if print_diagnostic:
            print('--- :/ TAI NOT CREATED BEFORE {}'.format(variables['MAX_YEAR'] + 1))
            plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()

    return {'tai_year': int(tai_year),
            'delay': int(nonscaling_delay_out),
            'effective_flop': effective_flop_collector}


def print_graph(samples, label, reverse=False, digits=1):
    pctiles = sq.get_percentiles(samples, reverse=reverse, digits=digits)
    if len(set(samples)) > 1:
        print('## {} ##'.format(label))
        pprint(pctiles)
        print('(Mean: {})'.format(round(np.mean(samples), 1)))
        plt.hist(samples, bins = 200)
        plt.show()
        print('-')
        print('-')
    else:
        print('## {}: {} ##'.format(label, samples[0]))
        print('-')
    return pctiles


def print_tai_arrival_stats(tai_years, variables):
    print('## DISTRIBUTION OF TAI ARRIVAL DATE ##')
    pctiles = sq.get_percentiles(tai_years, percentiles=[5, 10, 15, 20, 25, 35, 50, 60, 75, 80, 90, 95])
    pprint([str(o[0]) + '%: ' + (str(int(o[1])) if o[1] < variables['MAX_YEAR'] else '>' + str(variables['MAX_YEAR'])) for o in pctiles.items()])
    print('(Mean: {})'.format(int(round(np.mean(tai_years)))))
    print('-')
    print('-')

    print('## DISTRIBUTION OF RELATIVE TAI ARRIVAL DATE ##')
    pprint([str(o[0]) + '%: ' + (str(int(o[1]) - variables['CURRENT_YEAR']) if o[1] < variables['MAX_YEAR'] else '>' + str(variables['MAX_YEAR'] - variables['CURRENT_YEAR'])) + ' years from now' for o in pctiles.items()])
    print('(Mean: {} years from now)'.format(int(round(np.mean([t - variables['CURRENT_YEAR'] for t in tai_years])))))
    print('-')
    print('-')


    print('## TAI ARRIVAL DATE BY BIN ##')

    def bin_tai_yrs(low=None, hi=None):
        low = variables['CURRENT_YEAR'] if low is None else low
        if hi is None:
            r = np.mean([y >= low for y in tai_years])
        else:
            r = np.mean([(y >= low) and (y <= hi) for y in tai_years])
        return round(r * 100, 1)


    print('This year: {}%'.format(bin_tai_yrs(hi=variables['CURRENT_YEAR'])))
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
    print('-')

    print('## TAI ARRIVAL DATE BY YEAR ##')
    print('By EOY 2024: {}%'.format(bin_tai_yrs(hi=2024)))
    print('By EOY 2025: {}%'.format(bin_tai_yrs(hi=2025)))
    print('By EOY 2027: {}% (within 5 yrs)'.format(bin_tai_yrs(hi=2027)))
    print('By EOY 2032: {}% (within 10yrs)'.format(bin_tai_yrs(hi=2032)))
    print('By EOY 2040: {}%'.format(bin_tai_yrs(hi=2040)))
    print('By EOY 2047: {}% (within 25yrs)'.format(bin_tai_yrs(hi=2047)))
    print('By EOY 2050: {}%'.format(bin_tai_yrs(hi=2050)))
    print('By EOY 2060: {}%'.format(bin_tai_yrs(hi=2060)))
    print('By EOY 2070: {}%'.format(bin_tai_yrs(hi=2070)))
    print('By EOY 2072: {}% (within 50yrs)'.format(bin_tai_yrs(hi=2072)))
    print('By EOY 2078: {}% (within my expected lifetime)'.format(bin_tai_yrs(hi=2078)))
    print('By EOY 2100: {}%'.format(bin_tai_yrs(hi=2100)))
    print('By EOY 2122: {}%'.format(bin_tai_yrs(hi=2122)))
    print('-')
    print('-')
    
    print('## TAI ARRIVAL DATE BY YEAR - COMPARE TO AJEYA 2020 BENCHMARK ##')
    print('By EOY 2028 - this model {}% vs. Ajeya 2020 5%'.format(bin_tai_yrs(hi=2028)))
    print('By EOY 2032 - this model {}% vs. Ajeya 2020 10%'.format(bin_tai_yrs(hi=2032)))
    print('By EOY 2035 - this model {}% vs. Ajeya 2020 15%'.format(bin_tai_yrs(hi=2035)))
    print('By EOY 2040 - this model {}% vs. Ajeya 2020 25%'.format(bin_tai_yrs(hi=2040)))
    print('By EOY 2053 - this model {}% vs. Ajeya 2020 50%'.format(bin_tai_yrs(hi=2053)))
    print('By EOY 2062 - this model {}% vs. Ajeya 2020 60%'.format(bin_tai_yrs(hi=2062)))
    print('By EOY 2084 - this model {}% vs. Ajeya 2020 75%'.format(bin_tai_yrs(hi=2084)))
    print('By EOY 2100 - this model {}% vs. Ajeya 2020 78%'.format(bin_tai_yrs(hi=2100)))
    
    print('-')
    print('-')
    print('## TAI ARRIVAL DATE BY YEAR - COMPARE TO AJEYA 2022 BENCHMARK ##')
    print('By EOY 2030 - this model {}% vs. Ajeya 2022 15%'.format(bin_tai_yrs(hi=2030)))
    print('By EOY 2036 - this model {}% vs. Ajeya 2022 35%'.format(bin_tai_yrs(hi=2036)))
    print('By EOY 2040 - this model {}% vs. Ajeya 2022 50%'.format(bin_tai_yrs(hi=2040)))
    print('By EOY 2050 - this model {}% vs. Ajeya 2022 60%'.format(bin_tai_yrs(hi=2050)))
    
    print('-')
    print('-')
    print('## TAI ARRIVAL DATE BY YEAR - COMPARE TO EPOCH 2023 BENCHMARK ##')
    print('By EOY 2030 - this model {}% vs. Epoch 2023 20%'.format(bin_tai_yrs(hi=2030)))
    print('By EOY 2050 - this model {}% vs. Epoch 2023 51%'.format(bin_tai_yrs(hi=2050)))
    print('By EOY 2100 - this model {}% vs. Epoch 2023 61%'.format(bin_tai_yrs(hi=2100)))

    tai_years_ = np.array([variables['MAX_YEAR'] + 1 if t > variables['MAX_YEAR'] else t for t in tai_years])
    count, bins_count = np.histogram(tai_years_, bins=(variables['MAX_YEAR'] - variables['CURRENT_YEAR']))
    bins = np.round(np.array([b for b in bins_count[1:] if b <= variables['MAX_YEAR']]))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    plt.plot(bins, cdf[:len(bins)], label='CDF')
    if variables['MAX_YEAR'] > 2200:
        plt.xlim([variables['CURRENT_YEAR'], 2200])
    plt.legend()
    plt.show()

    pdf_smoothed = savitzky_golay(pdf[:len(bins)], 51, 3)
    plt.plot(bins, pdf_smoothed, label='PDF (smoothed)')
    if variables['MAX_YEAR'] > 2200:
        plt.xlim([variables['CURRENT_YEAR'], 2200])
    plt.legend()
    plt.show()


def define_tai_timeline_event(variables, verbose=False):
    tai_flop_size_ = variables['tai_flop_size']
    if sq.is_sampleable(tai_flop_size_):
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
    gdp_growth_ = sq.sample(variables['gdp_growth'])
    max_gdp_frac_ = sq.sample(variables['max_gdp_frac'])

    initial_pay_ = variables['initial_pay']
    if sq.is_sampleable(initial_pay_):
        initial_pay_ = sq.sample(initial_pay_)
    else:
        initial_pay_ = sq.sample(sq.discrete(variables['initial_pay']))
    initial_pay_ = 10 ** initial_pay_
    
    willingness_ramp_happens = sq.event_occurs(variables.get('p_willingness_ramp', 0))
    if willingness_ramp_happens:
        willingness_ramp_ = sq.sample(variables.get('willingness_ramp', 1))
    else:
        willingness_ramp_ = 1
    
    initial_gdp_ = variables['initial_gdp']
    spend_doubling_time_ = sq.sample(variables['spend_doubling_time'])
    spend_doubling_time_2025_ = sq.sample(variables['2025_spend_doubling_time']) if variables.get('2025_spend_doubling_time') else spend_doubling_time_
    nonscaling_delay_ = variables.get('nonscaling_delay')
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
                               spend_doubling_time_2025_=spend_doubling_time_2025_,
                               nonscaling_delay_=nonscaling_delay_,
                               willingness_spend_horizon_=willingness_spend_horizon_,
                               variables=variables,
                               print_diagnostic=verbose)


def run_timelines_model(variables, cores=1, runs=10000, load_cache_file=None,
                        dump_cache_file=None):
    for i in range(3):
        print('-')
        print('-')
        print('## SAMPLE RUN {} ##'.format(i + 1))
        define_tai_timeline_event(variables, verbose=True)

    print('-')
    print('-')
    print('## RUN TIMELINES MODEL ##')
    tai_years = bayes.bayesnet(lambda: define_tai_timeline_event(variables),
                               verbose=True,
                               raw=True,
                               cores=cores,
                               load_cache_file=load_cache_file,
                               dump_cache_file=dump_cache_file,
                               n=runs)

    print('-')
    print_tai_arrival_stats([t['tai_year'] for t in tai_years], variables)
    print('-')
    print('-')

    if variables['MAX_YEAR'] > 2200:
        years = list(range(variables['CURRENT_YEAR'], 2200))

    initial_flop_s = variables['tai_flop_size']
    if sq.is_sampleable(initial_flop_s):
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

    initial_pay_s = variables['initial_pay']
    if sq.is_sampleable(initial_pay_s):
        initial_pay_s = sq.sample(initial_pay_s, n=1000)
    else:
        initial_pay_s = sq.sample(sq.discrete(initial_pay_s), n=1000)
    initial_pay_p = print_graph(initial_pay_s, 'INITIAL PAY')

    gdp_growth_s = sq.sample(variables['gdp_growth'], n=1000)
    gdp_growth_p = print_graph(gdp_growth_s, label='GDP GROWTH', digits=2)

    max_gdp_frac_s = sq.sample(variables['max_gdp_frac'], n=1000)
    max_gdp_frac_p = print_graph(max_gdp_frac_s, label='MAX GDP FRAC', digits=5)

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
                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    gdp_10 = np.array([gdp(initial_gdp=variables['initial_gdp'],
                           gdp_growth=gdp_growth_p[10],
                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    gdp_90 = np.array([gdp(initial_gdp=variables['initial_gdp'],
                           gdp_growth=gdp_growth_p[90],
                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    plt.plot(years, np.log10(gdp_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(gdp_50), color='black')
    plt.plot(years, np.log10(gdp_90), linestyle='dashed', color='black')
    plt.ylabel('log GDP')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - GDP log 2022$USD {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(gdp_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(gdp_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(gdp_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(gdp_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(gdp_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(gdp_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Willingness to Pay Over Time ##')
    willingness_50 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[50],
                                                  initial_pay=10 ** initial_pay_p[50],
                                                  spend_doubling_time=spend_doubling_time_p[50],
                                                  max_gdp_frac=max_gdp_frac_p[50],
                                                  year=(y - variables['CURRENT_YEAR'])) for y in years])
    willingness_10 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[10],
                                                  initial_pay=10 ** initial_pay_p[10],
                                                  spend_doubling_time=spend_doubling_time_p[10],
                                                  max_gdp_frac=max_gdp_frac_p[10],
                                                  year=(y - variables['CURRENT_YEAR'])) for y in years])
    willingness_90 = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                  gdp_growth=gdp_growth_p[90],
                                                  initial_pay=10 ** initial_pay_p[90],
                                                  spend_doubling_time=spend_doubling_time_p[90],
                                                  max_gdp_frac=max_gdp_frac_p[90],
                                                  year=(y - variables['CURRENT_YEAR'])) for y in years])

    plt.plot(years, np.log10(willingness_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(willingness_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(willingness_50), color='black')
    plt.ylabel('log 2022$USD/yr willing to spend on TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - willingness log 2022$USD per year {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(willingness_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(willingness_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(willingness_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(willingness_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(willingness_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(willingness_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Actual FLOP Needed to Make TAI (Given Algorithmic Progress) ##')
    flops_50 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[50],
                                     doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                   algo_doubling_rate_max_p[50],
                                                                   initial_flop_p[50]),
                                     possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[50],
                                                                                         max_reduction_p[50],
                                                                                         initial_flop_p[50]),
                                     year=(y - variables['CURRENT_YEAR'])) for y in years])
    flops_10 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[10],
                                     doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                   algo_doubling_rate_max_p[10],
                                                                   initial_flop_p[10]),
                                     possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[10],
                                                                                         max_reduction_p[10],
                                                                                         initial_flop_p[10]),
                                     year=(y - variables['CURRENT_YEAR'])) for y in years])
    flops_90 = np.array([flop_needed(initial_flop=10 ** initial_flop_p[90],
                                     doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                   algo_doubling_rate_max_p[90],
                                                                   initial_flop_p[90]),
                                     possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[90],
                                                                                         max_reduction_p[90],
                                                                                         initial_flop_p[90]),
                                     year=(y - variables['CURRENT_YEAR'])) for y in years])

    plt.plot(years, np.log10(flops_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flops_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flops_50), color='black')
    plt.ylabel('log actual FLOP needed to make TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - log FLOP needed for TAI {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flops_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flops_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flops_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flops_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flops_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flops_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Actual FLOP per Dollar (Given Declining Costs) ##')
    flop_per_dollar_50 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                                   max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                                   halving_rate=flop_halving_rate_p[50],
                                                   year=(y - variables['CURRENT_YEAR'])) for y in years])
    flop_per_dollar_90 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                                   max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                                   halving_rate=flop_halving_rate_p[90],
                                                   year=(y - variables['CURRENT_YEAR'])) for y in years])
    flop_per_dollar_10 = np.array([flop_per_dollar(initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                                   max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                                   halving_rate=flop_halving_rate_p[10],
                                                   year=(y - variables['CURRENT_YEAR'])) for y in years])
    plt.plot(years, np.log10(flop_per_dollar_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_per_dollar_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_per_dollar_50), color='black')
    plt.ylabel('log FLOP per $1')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - log FLOP per 2022$1USD {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flop_per_dollar_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_per_dollar_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flop_per_dollar_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_per_dollar_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flop_per_dollar_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_per_dollar_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Max Possible OOM Reduction in TAI FLOP Size ##')
    tai_sizes = range(20, 50)
    algo_reduction_50 = np.array([possible_algo_reduction_fn(min_reduction_p[50],
                                                              max_reduction_p[50], t) for t in tai_sizes])
    algo_reduction_10 = np.array([possible_algo_reduction_fn(min_reduction_p[10],
                                                              max_reduction_p[90], t) for t in tai_sizes])
    algo_reduction_90 = np.array([possible_algo_reduction_fn(min_reduction_p[90],
                                                              max_reduction_p[10], t) for t in tai_sizes])
    plt.plot(tai_sizes, algo_reduction_50, color='black')
    plt.plot(tai_sizes, algo_reduction_10, linestyle='dashed', color='black')
    plt.plot(tai_sizes, algo_reduction_90, linestyle='dashed', color='black')
    plt.ylabel('max OOM reduction')
    plt.xlabel('initial FLOP needed for TAI prior to any reduction')
    plt.show()

    for t in tai_sizes:
        print('TAI log FLOP {} -> {} OOM reductions possible (90% CI: {} to {})'.format(t,
                                                                                        round(possible_algo_reduction_fn(min_reduction_p[50],
                                                                                                                         max_reduction_p[50],
                                                                                                                         t), 2),
                                                                                        round(possible_algo_reduction_fn(min_reduction_p[10],
                                                                                                                         max_reduction_p[90],
                                                                                                                         t), 2),
                                                                                        round(possible_algo_reduction_fn(min_reduction_p[90],
                                                                                                                         max_reduction_p[10],
                                                                                                                         t), 2)))


    print('-')
    print('-')
    print('## Halving time (years) of compute requirements ##')
    halving_time_50 = np.array([algo_halving_fn(algo_doubling_rate_min_p[50],
                                                algo_doubling_rate_max_p[50],
                                                t) for t in tai_sizes])
    halving_time_90 = np.array([algo_halving_fn(algo_doubling_rate_min_p[90],
                                                algo_doubling_rate_max_p[90],
                                                t) for t in tai_sizes])
    halving_time_10 = np.array([algo_halving_fn(algo_doubling_rate_min_p[10],
                                                algo_doubling_rate_max_p[10],
                                                t) for t in tai_sizes])
    plt.plot(tai_sizes, halving_time_50, color='black')
    plt.plot(tai_sizes, halving_time_90, linestyle='dashed', color='black')
    plt.plot(tai_sizes, halving_time_10, linestyle='dashed', color='black')
    plt.ylabel('number of years for compute requirements to halve')
    plt.show()

    for t in tai_sizes:
        print('TAI log FLOP {} -> algo doubling rate {}yrs (90% CI: {} to {})'.format(t,
                                                                                      round(algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                                            algo_doubling_rate_max_p[50],
                                                                                            t), 2),
                                                                                      round(algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                                            algo_doubling_rate_max_p[90],
                                                                                            t), 2),
                                                                                      round(algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                                            algo_doubling_rate_max_p[10],
                                                                                            t), 2)))


    if variables.get('initial_chance_of_nonscaling_issue', 0) != 0:
        p_delay_ = np.array([p_nonscaling_delay(y) for y in years])
        plt.plot(years, p_delay_, color='black')
        plt.ylabel('chance of a non-scaling delay')
        plt.show()

        for y in years[:10] + years[10::10]:
            outstr = 'Year: {} - chance of a nonscaling delay if TAI compute needs are otherwise met in this year: {}%'
            print(outstr.format(y, int(round(p_delay_[y - variables['CURRENT_YEAR']] * 100))))


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
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    cost_of_tai_10 = np.array([cost_of_tai(initial_flop=10 ** initial_flop_p[10],
                                           possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[10], max_reduction_p[10], initial_flop_p[10]),
                                           algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                              algo_doubling_rate_max_p[10],
                                                                              initial_flop_p[10]),
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                           flop_halving_rate=flop_halving_rate_p[10],
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    cost_of_tai_90 = np.array([cost_of_tai(initial_flop=10 ** initial_flop_p[90],
                                           possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[90], max_reduction_p[90], initial_flop_p[90]),
                                           algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                              algo_doubling_rate_max_p[90],
                                                                              initial_flop_p[90]),
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                           flop_halving_rate=flop_halving_rate_p[90],
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])

    plt.plot(years, np.log10(cost_of_tai_50), color='black')
    plt.plot(years, np.log10(cost_of_tai_10), linestyle='dashed', color='black')
    plt.plot(years, np.log10(cost_of_tai_90), linestyle='dashed', color='black')
    plt.ylabel('log $ needed to buy TAI')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - {} log 2022$USD to buy TAI (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(cost_of_tai_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(cost_of_tai_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(cost_of_tai_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(cost_of_tai_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(cost_of_tai_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(cost_of_tai_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Actual FLOP at Max Spend ##')
    flop_at_max_50 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                           gdp_growth=gdp_growth_p[50],
                                           initial_pay=10 ** initial_pay_p[50],
                                           spend_doubling_time=spend_doubling_time_p[50],
                                           max_gdp_frac=max_gdp_frac_p[50],
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                           flop_halving_rate=flop_halving_rate_p[50],
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    flop_at_max_10 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                           gdp_growth=gdp_growth_p[10],
                                           initial_pay=10 ** initial_pay_p[10],
                                           spend_doubling_time=spend_doubling_time_p[10],
                                           max_gdp_frac=max_gdp_frac_p[10],
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                           flop_halving_rate=flop_halving_rate_p[10],
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])
    flop_at_max_90 = np.array([flop_at_max(initial_gdp=variables['initial_gdp'],
                                           gdp_growth=gdp_growth_p[90],
                                           initial_pay=10 ** initial_pay_p[90],
                                           spend_doubling_time=spend_doubling_time_p[90],
                                           max_gdp_frac=max_gdp_frac_p[90],
                                           initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                           max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                           flop_halving_rate=flop_halving_rate_p[90],
                                           year=(y - variables['CURRENT_YEAR'])) for y in years])

    plt.plot(years, np.log10(flop_at_max_50), color='black')
    plt.plot(years, np.log10(flop_at_max_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(flop_at_max_10), linestyle='dashed', color='black')
    plt.ylabel('max log FLOP bought given willingness to spend')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - max log FLOP {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(flop_at_max_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_at_max_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flop_at_max_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_at_max_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(flop_at_max_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(flop_at_max_90[y - variables['CURRENT_YEAR']])))


    print('-')
    print('-')
    print('## Effective 2023-FLOP at Max Spend (given algorithmic progress and decline in $/FLOP) ##')
    effective_flop_at_max_50 = np.array([effective_flop_at_max(initial_gdp=variables['initial_gdp'],
                                                               gdp_growth=gdp_growth_p[50],
                                                               initial_pay=10 ** initial_pay_p[50],
                                                               spend_doubling_time=spend_doubling_time_p[50],
                                                               max_gdp_frac=max_gdp_frac_p[50],
                                                               initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[50],
                                                               initial_flop=10 ** initial_flop_p[50],
                                                               max_flop_per_dollar=10 ** max_flop_per_dollar_p[50],
                                                               flop_halving_rate=flop_halving_rate_p[50],
                                                               possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[50],
                                                                                                                   max_reduction_p[50],
                                                                                                                   initial_flop_p[50]),
                                                               doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[50],
                                                                                             algo_doubling_rate_max_p[50],
                                                                                             initial_flop_p[50]),
                                                               year=(y - variables['CURRENT_YEAR'])) for y in years])
    effective_flop_at_max_10 = np.array([effective_flop_at_max(initial_gdp=variables['initial_gdp'],
                                                               gdp_growth=gdp_growth_p[10],
                                                               initial_pay=10 ** initial_pay_p[10],
                                                               spend_doubling_time=spend_doubling_time_p[10],
                                                               max_gdp_frac=max_gdp_frac_p[10],
                                                               initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[10],
                                                               initial_flop=10 ** initial_flop_p[10],
                                                               max_flop_per_dollar=10 ** max_flop_per_dollar_p[10],
                                                               flop_halving_rate=flop_halving_rate_p[10],
                                                               possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[90],
                                                                                                                   max_reduction_p[10],
                                                                                                                   initial_flop_p[10]),
                                                               doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[90],
                                                                                             algo_doubling_rate_max_p[10],
                                                                                             initial_flop_p[10]),
                                                               year=(y - variables['CURRENT_YEAR'])) for y in years])
    effective_flop_at_max_90 = np.array([effective_flop_at_max(initial_gdp=variables['initial_gdp'],
                                                               gdp_growth=gdp_growth_p[90],
                                                               initial_pay=10 ** initial_pay_p[90],
                                                               spend_doubling_time=spend_doubling_time_p[90],
                                                               max_gdp_frac=max_gdp_frac_p[90],
                                                               initial_flop_per_dollar=10 ** initial_flop_per_dollar_p[90],
                                                               initial_flop=10 ** initial_flop_p[90],
                                                               max_flop_per_dollar=10 ** max_flop_per_dollar_p[90],
                                                               flop_halving_rate=flop_halving_rate_p[90],
                                                               possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[10],
                                                                                                                   max_reduction_p[90],
                                                                                                                   initial_flop_p[90]),
                                                               doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[10],
                                                                                             algo_doubling_rate_max_p[90],
                                                                                             initial_flop_p[90]),
                                                               year=(y - variables['CURRENT_YEAR'])) for y in years])

    plt.plot(years, np.log10(effective_flop_at_max_50), color='black')
    plt.plot(years, np.log10(effective_flop_at_max_90), linestyle='dashed', color='black')
    plt.plot(years, np.log10(effective_flop_at_max_10), linestyle='dashed', color='black')
    plt.ylabel('max log effective 2023-FLOP bought')
    plt.show()

    for y in years[:10] + years[10::10]:
        outstr = 'Year: {} - max log effective 2023-FLOP {} (~{}) 90% CI {} (~{}) - {} (~{})'
        print(outstr.format(y,
                            np.round(np.log10(effective_flop_at_max_50[y - variables['CURRENT_YEAR']]), 1),
                            numerize(effective_flop_at_max_50[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(effective_flop_at_max_10[y - variables['CURRENT_YEAR']]), 1),
                            numerize(effective_flop_at_max_10[y - variables['CURRENT_YEAR']]),
                            np.round(np.log10(effective_flop_at_max_90[y - variables['CURRENT_YEAR']]), 1),
                            numerize(effective_flop_at_max_90[y - variables['CURRENT_YEAR']])))


    nonscaling_delay_ = variables.get('nonscaling_delay')
    if nonscaling_delay_ is not None:
        print('-')
        print('-')
        print('## Nonscaling delay ##')
        if len(nonscaling_delay_) > 1:
            print('There are {} ways a non-scaling delay could happen.'.format(len(nonscaling_delay_)))
            for name, delay in nonscaling_delay_.items():
                print('- {}: additional {} years if it happens'.format(name, delay['length']))
                pprint(sq.get_percentiles(sq.sample(delay['length'], n=1000), digits=0))
                plot_nonscaling_delay(plt, years, delay['prob'])
        else:
            delay = list(nonscaling_delay_.items())[0][1]
            print(('If a non-scaling delay happens, it will take an additional {} years to produce TAI due' +
                   ' to issues unrelated to scaling FLOP').format(delay['length']))
            pprint(sq.get_percentiles(sq.sample(delay['length'], n=1000)))
            plot_nonscaling_delay(plt, years, delay['prob'])


    print('-')
    print('-')
    print('## Aggregate nonscaling delay probability ##')

    def bin_tai_delay_by_year(tai_years, low=None, hi=None):
        low = variables['CURRENT_YEAR'] if low is None else low
        if hi is None:
            tai_years = [y for y in tai_years if y['tai_year'] >= low]
        else:
            tai_years = [y for y in tai_years if (y['tai_year'] >= low) and (y['tai_year'] <= hi)]
        return int(round(np.mean([y['delay'] > 0 for y in tai_years]) * 100))

    print('If TAI compute level achieved in 2023-2026... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2023, 2026)))
    print('If TAI compute level achieved in 2027-2030... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2027, 2030)))
    print('If TAI compute level achieved in 2031-2035... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2031, 2035)))
    print('If TAI compute level achieved in 2036-2040... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2036, 2040)))
    print('If TAI compute level achieved in 2041-2050... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2041, 2050)))
    print('If TAI compute level achieved in 2051-2060... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2051, 2060)))
    print('If TAI compute level achieved in 2061-2100... {}% chance of TAI nonscaling delay'.format(bin_tai_delay_by_year(tai_years, 2061, 2100)))

    print('-')
    print('-')
    print('## Aggregate nonscaling delay length ##')
    delay_samples = [t['delay'] for t in tai_years]
    pprint(sq.get_percentiles(delay_samples, digits=0))
    plt.hist(delay_samples, bins=200)
    plt.xlabel('total years of delay')
    plt.show()

    return None
