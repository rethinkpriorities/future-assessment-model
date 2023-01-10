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


def flops_needed(initial_flops, possible_reduction, doubling_rate, year):
    x = (np.log(2) / doubling_rate) * year
    if x < 650:
        y = (math.log10(initial_flops) - max(math.log10(math.exp(x)) - math.log10(1 + (1/possible_reduction) * math.exp(x)), 0))
        if y > 300:
            y = int(y) # Handle overflow errors        
        return 10 ** y
    else: # Handle math.exp and math.log10 overflow errors
        return 10 ** int(math.log10(initial_flops) - (1/possible_reduction))

    
def flops_per_dollar(initial_flops_per_dollar, max_flops_per_dollar, halving_rate, year):
    x = (np.log(2) / halving_rate) * year
    if x < 650:
        y = (math.log10(initial_flops_per_dollar) + math.log10(math.exp(x)) - math.log10(1 + initial_flops_per_dollar / max_flops_per_dollar * math.exp(x)))
        if y > 300:
            y = int(y) # Handle overflow errors                
        return 10 ** y
    else: # Handle math.exp and math.log10 overflow errors
        return 10 ** int(math.log10(initial_flops_per_dollar) + (year/halving_rate)/3.3)

    
def cost_of_tai(initial_flops, possible_reduction, algo_doubling_rate, initial_flops_per_dollar, max_flops_per_dollar,
                flops_halving_rate, year):
    return (flops_needed(initial_flops, possible_reduction, algo_doubling_rate, year) /
            flops_per_dollar(initial_flops_per_dollar, max_flops_per_dollar, flops_halving_rate, year))


def flops_at_max(initial_gdp, gdp_growth, initial_pay, spend_doubling_time, max_gdp_frac,
                 initial_flops_per_dollar, max_flops_per_dollar, flops_halving_rate, year):
    return (willingness_to_pay(initial_gdp=initial_gdp,
                               gdp_growth=gdp_growth,
                               initial_pay=initial_pay,
                               spend_doubling_time=spend_doubling_time,
                               max_gdp_frac=max_gdp_frac,
                               year=year) *
            flops_per_dollar(initial_flops_per_dollar, max_flops_per_dollar, flops_halving_rate, year))


def possible_algo_reduction_fn(min_reduction, max_reduction, tai_flop_size):
    if max_reduction < min_reduction:
        max_reduction = min_reduction
    if min_reduction > max_reduction:
        min_reduction = max_reduction
    return min(max(min_reduction + round((tai_flop_size - 32) / 4), min_reduction), max_reduction)


def p_nonscaling_delay(initial_p, final_p, year, max_year):
    return generalized_logistic_curve(x=year - CURRENT_YEAR,
                                      slope=0.3,
                                      shift=3 * (max_year - CURRENT_YEAR),
                                      push=1,
                                      maximum=final_p,
                                      minimum=initial_p)


def run_tai_model_round(initial_gdp_, tai_flop_size_, nonscaling_delay_, algo_doubling_rate_,
                        possible_algo_reduction_, initial_flops_per_dollar_,
                        flops_halving_rate_, max_flops_per_dollar_, initial_pay_, gdp_growth_,
                        max_gdp_frac_, willingness_ramp_, spend_doubling_time_,
                        initial_chance_of_nonscaling_issue_, final_chance_of_nonscaling_issue_,
                        nonscaling_issue_bottom_year_, willingness_spend_horizon_,
                        print_diagnostic):
    queue_tai_year = 99999
    plt.ioff()
    if print_diagnostic:
        cost_of_tai_collector = []
        willingness_collector = []
    
    if print_diagnostic:
        print('It takes {} log FLOPs (~{}) for transformative capabilities.'.format(np.round(tai_flop_size_, 1),
                                                                                    numerize(10 ** tai_flop_size_)))
        print('Every {} years algorithms get 2x better, with {} log reductions possible.'.format(np.round(algo_doubling_rate_, 1),
                                                                                                 np.round(possible_algo_reduction_, 1)))
        print(('FLOPs start at a cost of {} log FLOPs (~{}) per 2022$USD. Every {} years they get ' +
               '2x cheaper, to a maximum of {} log FLOPs (~{}) per 2022$USD.').format(np.round(math.log10(initial_flops_per_dollar_), 1),
                                                                               numerize(initial_flops_per_dollar_),
                                                                               np.round(flops_halving_rate_, 1),
                                                                               np.round(math.log10(max_flops_per_dollar_), 1),
                                                                               numerize(max_flops_per_dollar_)))
        print(('We are willing to pay {} log 2022$USD (~{}) and this increases by {}x per year to a max of {}% of GDP. ' +
               'GDP grows at a rate of {}x per year.').format(np.round(math.log10(initial_pay_), 1),
                                                              numerize(initial_pay_),
                                                              np.round(spend_doubling_time_, 1),
                                                              np.round(max_gdp_frac_, 4),
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
            flops_needed_ = flops_needed(initial_flops=10 ** tai_flop_size_,
                                         doubling_rate=algo_doubling_rate_,
                                         possible_reduction=10 ** possible_algo_reduction_,
                                         year=(y - CURRENT_YEAR))
            
            flops_per_dollar_ = flops_per_dollar(initial_flops_per_dollar=initial_flops_per_dollar_,
                                                 max_flops_per_dollar=max_flops_per_dollar_,
                                                 halving_rate=flops_halving_rate_,
                                                 year=(y - CURRENT_YEAR))
            
            if flops_per_dollar_ > 10 ** 200 or flops_needed_ > 10 ** 200:
                flops_needed_ = int(flops_needed_)
                flops_per_dollar_ = int(flops_per_dollar_)
                cost_of_tai_ = flops_needed_ // flops_per_dollar_
            else:
                cost_of_tai_ = flops_needed_ / flops_per_dollar_
            
            willingness_ = willingness_to_pay(initial_gdp=initial_gdp_,
                                              gdp_growth=gdp_growth_,
                                              initial_pay=initial_pay_,
                                              spend_doubling_time=spend_doubling_time_,
                                              max_gdp_frac=max_gdp_frac_,
                                              year=(y - CURRENT_YEAR))
            
            if flops_per_dollar_ > 10 ** 200:
                willingness_ = int(willingness_)
            if willingness_ > 10 ** 200:
                flops_per_dollar_ = int(flops_per_dollar_)
            
            if print_diagnostic:
                cost_of_tai_collector.append(cost_of_tai_)
                willingness_collector.append(willingness_)
            
            total_compute_ = willingness_ * flops_per_dollar_
            
            if print_diagnostic:
                out_str = ('Year: {} - {} max log FLOP ({}) available - TAI takes {} log FLOP ({}) - ' +
                           'log $ {} to buy TAI ({}) vs. willingness to pay log $ {} ({}) - {} log FLOPS per $ ({})')
                print(out_str.format(y,
                                     np.round(math.log10(total_compute_), 1),
                                     numerize(total_compute_),
                                     np.round(math.log10(flops_needed_), 1),
                                     numerize(flops_needed_),
                                     np.round(math.log10(cost_of_tai_), 1),
                                     numerize(cost_of_tai_),
                                     np.round(math.log10(willingness_), 1),
                                     numerize(willingness_),
                                     np.round(math.log10(flops_per_dollar_), 1),
                                     numerize(flops_per_dollar_)))
            
            if cost_of_tai_ > 10 ** 200:
                spend_tai_years = int(cost_of_tai_) // int(willingness_)
            else:
                spend_tai_years = cost_of_tai_ / willingness_
                
            if not is_nonscaling_issue and queue_tai_year < 99999 and print_diagnostic:
                print('-$- {}/{}'.format(y, queue_tai_year))
            if (cost_of_tai_ * willingness_ramp_) <= willingness_ or y >= queue_tai_year:
                if is_nonscaling_issue is None:
                    p_nonscaling_delay_ = p_nonscaling_delay(initial_chance_of_nonscaling_issue_,
                                                             final_chance_of_nonscaling_issue_,
                                                             year=y,
                                                             max_year=nonscaling_issue_bottom_year_)
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
                        print('/!\ FLOPs for TAI sufficient but needs {} more years to solve non-scaling issues'.format(np.round(nonscaling_countdown, 1)))
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


def print_graph(samples, label, reverse=False):
    if len(set(samples)) > 1:
        print('## {} ##'.format(label))
        pprint(sq.get_percentiles(samples, reverse=reverse, digits=1))
        plt.hist(samples, bins = 200)
        plt.show()
        print('-')
        print('-')
    else:
        print('## {}: {} ##'.format(label, samples[0]))
        print('-')
    return None


def run_timelines_model(variables, cores=1, runs=10000, load_cache_file=None,
                        dump_cache_file=None, reload_cache=False):
    initial_flop_size_s = variables['tai_flop_size']
    print_graph(initial_flop_size_s, 'TAI FLOP SIZE')

    min_reduction_s = sq.sample(variables['min_reduction'], n=1000)
    print_graph(min_reduction_s, 'MIN REDUCTION')

    max_reduction_s = sq.sample(variables['max_reduction'], n=1000)
    print_graph(max_reduction_s, 'MAX REDUCTION', reverse=True)

    algo_doubling_rate_min_s = sq.sample(variables['algo_doubling_rate_min'], n=1000)
    print_graph(algo_doubling_rate_min_s, label='MIN ALGO DOUBLING RATE', reverse=True)

    algo_doubling_rate_max_s = sq.sample(variables['algo_doubling_rate_max'], n=1000)
    print_graph(algo_doubling_rate_max_s, label='MAX ALGO DOUBLING RATE', reverse=True)

    initial_flops_per_dollar_s = sq.sample(variables['initial_flops_per_dollar'], n=1000)
    print_graph(initial_flops_per_dollar_s, label='INITIAL FLOPS PER DOLLAR')

    flops_halving_rate_s = sq.sample(variables['flops_halving_rate'], n=1000)
    print_graph(flops_halving_rate_s, label='FLOPS HALVING RATE', reverse=True)

    max_flops_per_dollar_s = sq.sample(variables['max_flops_per_dollar'], n=1000)
    print_graph(max_flops_per_dollar_s, label='MAX FLOPS PER DOLLAR')

    initial_pay_s = sq.sample(variables['initial_pay'], n=1000)
    print_graph(initial_pay_s, label='INITIAL PAY')

    gdp_growth_s = sq.sample(variables['gdp_growth'], n=1000)
    print_graph(gdp_growth_s, label='GDP GROWTH')

    max_gdp_frac_s = sq.sample(variables['max_gdp_frac'], n=1000)
    print_graph(max_gdp_frac_s, label='MAX GDP FRAC')

    willingness_ramp = variables.get('willingness_ramp', 0)
    if willingness_ramp != 0:
        willingness_ramp_s = sq.sample(willingness_ramp, n=1000)
        print_graph(willingness_ramp_s, label='WILLINGNESS RAMP')

    spend_doubling_time_s = sq.sample(variables['spend_doubling_time'], n=1000)
    print_graph(spend_doubling_time_s, label='SPEND DOUBLING TIME', reverse=True)

    willingness_spend_horizon = variables.get('willingness_spend_horizon', 1)
    if willingness_spend_horizon != 1:
        willingness_spend_horizon_s = sq.sample(willingness_spend_horizon, n=1000)
        print_graph(willingness_spend_horizon_s, label='WILLINGNESS SPEND HORIZON')


    """
## GDP Over Time
    gdp_ = np.array([gdp(initial_gdp=variables['initial_gdp'],
                         gdp_growth=gdp_growth_p[GRAPH_P],
                         year=(y - CURRENT_YEAR)) for y in years])
    plt.plot(years, np.log10(gdp_))
    plt.ylabel('log GDP')

    for y in years:
        print('Year: {} - GDP log 2022$USD {} (~{})'.format(y,
                                                            np.round(np.log10(gdp_[y - CURRENT_YEAR]), 1),
                                                            numerize(gdp_[y - CURRENT_YEAR])))

## Willingness to Pay Over Time
    for p in [20, 50, 80]:
        print('-')
        print('-')
        print('## {} ##'.format(p))
        willingness = np.array([willingness_to_pay(initial_gdp=variables['initial_gdp'],
                                                   gdp_growth=gdp_growth_p[p],
                                                   initial_pay=10 ** initial_pay_p[p],
                                                   spend_doubling_time=spend_doubling_time_p[p],
                                                   max_gdp_frac=max_gdp_frac_p[p],
                                                   year=(y - CURRENT_YEAR)) for y in years])
        for y in years:
            print('Year: {} - willingness log 2022$USD per year {} (~{})'.format(y,
                                                                                 np.round(np.log10(willingness[y - CURRENT_YEAR]), 1),
                                                                                 numerize(willingness[y - CURRENT_YEAR])))


        plt.plot(years, np.log10(willingness))
    plt.ylabel('log 2022$USD/yr willing to spend on TAI')
    plt.show()


## FLOPs Needed to Make TAI (Given Algorithmic Progress)
    flops_ = np.array([flops_needed(initial_flops=10 ** initial_flops_p[GRAPH_P],
                                    doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[GRAPH_P],
                                                                  algo_doubling_rate_max_p[GRAPH_P],
                                                                  initial_flops_p[GRAPH_P]),
                                    possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[GRAPH_P],
                                                                                        max_reduction_p[GRAPH_P],
                                                                                        initial_flops_p[GRAPH_P]),
                                    year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(flops_))
    plt.ylabel('log FLOPs needed to make TAI')

    for y in years:
        print('Year: {} - log FLOPs needed for TAI {} (~{})'.format(y,
                                                                    np.round(np.log10(flops_[y - CURRENT_YEAR]), 1),
                                                                    numerize(flops_[y - CURRENT_YEAR])))

## FLOPs per Dollar (Given Declining Costs)
    flops_per_dollar_ = np.array([flops_per_dollar(initial_flops_per_dollar=10 ** initial_flops_per_dollar_p[GRAPH_P],
                                                   max_flops_per_dollar=10 ** max_flops_per_dollar_p[GRAPH_P],
                                                   halving_rate=flops_halving_rate_p[GRAPH_P],
                                                   year=(y - CURRENT_YEAR)) for y in years])
    plt.plot(years, np.log10(flops_per_dollar_))
    plt.ylabel('log FLOPs per $1')

    for y in years:
        print('Year: {} - log {} FLOPs per $ (~{})'.format(y,
                                                           np.round(np.log10(flops_per_dollar_[y - CURRENT_YEAR]), 1),
                                                           numerize(flops_per_dollar_[y - CURRENT_YEAR])))

## Max Possible OOM Reduction in TAI FLOP Size

# TODO: Update to include efficiency based
    tai_sizes = range(24, 60)
    flops_per_dollar_ = np.array([possible_algo_reduction_fn(min_reduction_p[GRAPH_P],
                                                             max_reduction_p[GRAPH_P], t) for t in tai_sizes])
    plt.plot(tai_sizes, flops_per_dollar_)
    plt.ylabel('max OOM reduction')
    plt.xlabel('initial FLOP needed for TAI prior to any reduction')

    for t in tai_sizes:
        print('TAI log FLOP {} -> {} OOM reductions possible'.format(t,
                                                                     round(possible_algo_reduction_fn(min_reduction_p[GRAPH_P],
                                                                                                      max_reduction_p[GRAPH_P],
                                                                                                      t), 2)))


## Halving time (years) of compute requirements
    tai_sizes = range(24, 60)
    flops_per_dollar_ = np.array([algo_halving_fn(algo_doubling_rate_min_p[GRAPH_P],
                                                  algo_doubling_rate_max_p[GRAPH_P],
                                                  t) for t in tai_sizes])
    plt.plot(tai_sizes, flops_per_dollar_)
    plt.ylabel('number of years for compute requirements to halve')
    plt.xlabel('initial FLOP needed for TAI prior to any reduction')

    for t in tai_sizes:
        print('TAI log FLOP {} -> algo doubling rate {}yrs'.format(t,
                                                                   round(algo_halving_fn(algo_doubling_rate_min_p[GRAPH_P],
                                                                                         algo_doubling_rate_max_p[GRAPH_P],
                                                                                         t), 2)))


## Dollars Needed to Buy TAI (Given Algorithmic Progress and Decline in Cost per FLOP)

    cost_of_tai_ = np.array([cost_of_tai(initial_flops=10 ** initial_flops_p[GRAPH_P],
                                         possible_reduction=10 ** possible_algo_reduction_fn(min_reduction_p[GRAPH_P], max_reduction_p[GRAPH_P], initial_flops_p[GRAPH_P]),
                                         algo_doubling_rate=algo_halving_fn(algo_doubling_rate_min_p[GRAPH_P],
                                                                            algo_doubling_rate_max_p[GRAPH_P],
                                                                            initial_flops_p[GRAPH_P]),
                                         initial_flops_per_dollar=10 ** initial_flops_per_dollar_p[GRAPH_P],
                                         max_flops_per_dollar=10 ** max_flops_per_dollar_p[GRAPH_P],
                                         flops_halving_rate=flops_halving_rate_p[GRAPH_P],
                                         year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(cost_of_tai_))
    plt.ylabel('log $ needed to buy TAI')


    for y in years:
        print('Year: {} - log $ {} to buy TAI (~{})'.format(y,
                                                            np.round(np.log10(cost_of_tai_[y - CURRENT_YEAR]), 1),
                                                            numerize(cost_of_tai_[y - CURRENT_YEAR])))


## FLOPs at Max Spend

    flops_at_max_ = np.array([flops_at_max(initial_gdp=variables['initial_gdp'],
                                           gdp_growth=gdp_growth_p[GRAPH_P],
                                           initial_pay=10 ** initial_pay_p[GRAPH_P],
                                           spend_doubling_time=spend_doubling_time_p[GRAPH_P],
                                           max_gdp_frac=max_gdp_frac_p[GRAPH_P],
                                           initial_flops_per_dollar=10 ** initial_flops_per_dollar_p[GRAPH_P],
                                           max_flops_per_dollar=10 ** max_flops_per_dollar_p[GRAPH_P],
                                           flops_halving_rate=flops_halving_rate_p[GRAPH_P],
                                           year=(y - CURRENT_YEAR)) for y in years])

    plt.plot(years, np.log10(flops_at_max_))
    plt.ylabel('max log FLOPs bought given willingness to spend')


    for y in years:
        print('Year: {} - max log FLOPs {} (~{} FLOP, ~{} petaFLOP/s-days)'.format(y,
                                                                                   np.round(np.log10(flops_at_max_[y - CURRENT_YEAR]), 1),
                                                                                   numerize(flops_at_max_[y - CURRENT_YEAR]),
                                                                                   log_flop_to_petaflop_sdays(np.log10(flops_at_max_[y - CURRENT_YEAR]))))
        


## Parameters at Max Spend

    for y in years:
        print('Year: {} - max log parameters {} (~{} parameters)'.format(y,
                                                                         np.round(np.log10(flops_at_max_[y - CURRENT_YEAR]) - 12, 1),
                                                                         numerize(10 ** (np.log10(flops_at_max_[y - CURRENT_YEAR]) - 12))))
    """

    def define_event(verbose=False):
        tai_flop_size_ = sq.sample(sq.discrete(variables['tai_flop_size']))

        if tai_flop_size_ > 300:
            tai_flop_size_ = int(tai_flop_size_) # Handle overflow errors
        
        algo_doubling_rate_ = algo_halving_fn(sq.sample(variables['algo_doubling_rate_min']),
                                              sq.sample(variables['algo_doubling_rate_max']),
                                              tai_flop_size_)
        
        possible_algo_reduction_ = possible_algo_reduction_fn(sq.sample(variables['min_reduction']),
                                                              sq.sample(variables['max_reduction']),
                                                              tai_flop_size_)
        
        initial_flops_per_dollar_ = 10 ** sq.sample(variables['initial_flops_per_dollar'])
        flops_halving_rate_ = sq.sample(variables['flops_halving_rate'])
        max_flops_per_dollar_ = 10 ** sq.sample(variables['max_flops_per_dollar'])
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
        nonscaling_delay_ = sq.sample(variables.get('nonscaling_delay', sq.const(0)))
        initial_chance_of_nonscaling_issue_ = variables.get('initial_chance_of_nonscaling_issue', 0)
        final_chance_of_nonscaling_issue_ = variables.get('final_chance_of_nonscaling_issue', 0)
        nonscaling_issue_bottom_year_ = variables.get('nonscaling_issue_bottom_year', 0)
        willingness_spend_horizon_ = int(sq.sample(variables.get('willingness_spend_horizon', sq.const(1))))
        
        tai_year = run_tai_model_round(initial_gdp_=initial_gdp_,
                                       tai_flop_size_=tai_flop_size_,
                                       nonscaling_delay_=nonscaling_delay_,
                                       algo_doubling_rate_=algo_doubling_rate_,
                                       possible_algo_reduction_=possible_algo_reduction_,
                                       initial_flops_per_dollar_=initial_flops_per_dollar_,
                                       flops_halving_rate_=flops_halving_rate_,
                                       max_flops_per_dollar_=max_flops_per_dollar_,
                                       initial_pay_=initial_pay_,
                                       gdp_growth_=gdp_growth_,
                                       max_gdp_frac_=max_gdp_frac_,
                                       willingness_ramp_=willingness_ramp_,
                                       spend_doubling_time_=spend_doubling_time_,
                                       initial_chance_of_nonscaling_issue_=initial_chance_of_nonscaling_issue_,
                                       final_chance_of_nonscaling_issue_=final_chance_of_nonscaling_issue_,
                                       nonscaling_issue_bottom_year_=nonscaling_issue_bottom_year_,
                                       willingness_spend_horizon_=willingness_spend_horizon_,
                                       print_diagnostic=verbose)
        
        return {'tai_year': tai_year,
                'initial_gdp': initial_gdp_,
                'tai_flop_size': tai_flop_size_,
                'nonscaling_delay': nonscaling_delay_,
                'algo_doubling_rte': algo_doubling_rate_,
                'possible_algo_reduction': possible_algo_reduction_,
                'initial_flops_per_dollar': float(initial_flops_per_dollar_),
                'flops_halving_rate': flops_halving_rate_,
                'max_flops_per_dollar': float(max_flops_per_dollar_),
                'initial_pay': initial_pay_,
                'gdp_growth': gdp_growth_,
                'max_gdp_frac': max_gdp_frac_,
                'willingness_ramp_': willingness_ramp_,
                'spend_doubling_time_': spend_doubling_time_,
                'initial_chance_of_nonscaling_issue': initial_chance_of_nonscaling_issue_,
                'final_chance_of_nonscaling_issue': final_chance_of_nonscaling_issue_,
                'nonscaling_issue_bottom_year': nonscaling_issue_bottom_year_,
                'willingness_spend_horizon_': willingness_spend_horizon_}


    for i in range(3):
        print('## SAMPLE RUN {} ##'.format(i + 1))
        define_event(verbose=True)

    print('## FULL MODEL ##')
    tai_years = bayes.bayesnet(define_event,
                               find=lambda e: e['tai_year'],
                               verbose=True,
                               raw=True,
                               cores=cores,
                               load_cache_file=load_cache_file,
                               dump_cache_file=dump_cache_file,
                               reload_cache=reload_cache,
                               n=runs)

    out = sq.get_percentiles(tai_years)
    pprint([str(o[0]) + '%: ' + (str(int(o[1])) if o[1] < MAX_YEAR else '>' + str(MAX_YEAR)) for o in out.items()])
    print('-')
    print('-')

    # NOTE: Ajeya 2020's numbers should output something very close to:
    # '5%': 2027,
    # '10%: 2031',
    # '20%: 2037',
    # '30%: 2042',
    # '40%: 2047',
    # '50%: 2053',
    # '60%: 2061',
    # '70%: 2073',
    # '80%: >2100',
    # '90%: >2100',
    # '95%: >2100'

    pprint([str(o[0]) + '%: ' + (str(int(o[1]) - CURRENT_YEAR) if o[1] < MAX_YEAR else '>' + str(MAX_YEAR - CURRENT_YEAR)) + ' years from now' for o in out.items()])
    print('-')
    print('-')


    """
    tai_years_ = np.array([MAX_YEAR + 1 if t > MAX_YEAR else t for t in tai_years])
    count, bins_count = np.histogram(tai_years_, bins=(MAX_YEAR - CURRENT_YEAR))
    bins = np.round(np.array([b for b in bins_count[1:] if b <= MAX_YEAR]))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    plt.plot(bins, cdf[:len(bins)], label='CDF')
    plt.legend()
    plt.show()


    d_ = dict(zip(years, cdf[:len(bins)]))

    def bin_tai_yrs(low=None, hi=None):
        low = CURRENT_YEAR if low is None else low
        out = 1 - d_[low] if hi is None else d_[hi] - d_[low]
        return round(out * 100, 1)

    print('<2024: {}%'.format(bin_tai_yrs(hi=2024)))
    print('2025-2029: {}%'.format(bin_tai_yrs(2025, 2029)))
    print('2030-2039: {}%'.format(bin_tai_yrs(2030, 2039)))
    print('2040-2049: {}%'.format(bin_tai_yrs(2040, 2049)))
    print('2050-2059: {}%'.format(bin_tai_yrs(2050, 2059)))
    print('2060-2069: {}%'.format(bin_tai_yrs(2060, 2069)))
    print('2070-2079: {}%'.format(bin_tai_yrs(2070, 2079)))
    print('2080-2089: {}%'.format(bin_tai_yrs(2080, 2089)))
    print('2090-2099: {}%'.format(bin_tai_yrs(2090, 2099)))
    print('2100-2109: {}%'.format(bin_tai_yrs(2100, 2109)))
    print('2110-2119: {}%'.format(bin_tai_yrs(2110, 2119)))
    print('>2120: {}%'.format(bin_tai_yrs(low=2120)))


    pdf_smoothed = savitzky_golay(pdf[:len(bins)], 51, 3)
    plt.plot(bins, pdf_smoothed, label='PDF (smoothed)')
    plt.legend()
    plt.show()
    """
    return None
