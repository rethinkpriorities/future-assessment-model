STATES = ['boring', 'xrisk_tai_misuse', 'aligned_tai', 'xrisk_full_unaligned_tai_extinction',
          'xrisk_full_unaligned_tai_singleton', 'xrisk_subtly_unaligned_tai', 'xrisk_unknown_unknown',
          'xrisk_nanotech', 'xrisk_nukes_war', 'xrisk_nukes_accident', 'xrisk_bio_accident', 'xrisk_bio_war',
          'xrisk_bio_nonstate', 'xrisk_supervolcano']

extinctions = ['xrisk_full_unaligned_tai_extinction', 'xrisk_nukes_war', 'xrisk_nukes_accident',
               'xrisk_unknown_unknown', 'xrisk_nanotech', 'xrisk_bio_accident', 'xrisk_bio_war',
               'xrisk_bio_nonstate', 'xrisk_supervolcano']


def define_event(verbosity=0):
    state = {'category': 'boring', 'tai': False, 'tai_year': None, 'tai_type': None,
             'nano': False, 'wars': [], 'war': False, 'war_start_year': None,
             'war_end_year': None, 'russia_nuke_first': False, 'china_nuke_first': False,
             'war_belligerents': None, 'peace_until': None, 'engineered_pathogen': False,
             'natural_pathogen': False, 'averted_misalignment': False,
             'nuclear_weapon_used': False, 'catastrophe': [], 'recent_catastrophe_year': None,
             'terminate': False, 'final_year': None, 'double_catastrophe_xrisk': None}
    allowed_state_keys = list(state.keys())
    collectors = {}
    
    ## Set up timelines calculations
    # TODO: Can timelines stuff be factored out?
    if use_efficiency_based_algo_reduction:
        efficiency_ = sq.sample(efficiency)
        tai_flop_size_ = sq.sample(lambda: tai_flop_size(efficiency=sq.const(efficiency_),
                                                         debug=verbosity > 1))
    else:
        tai_flop_size_ = sq.sample(tai_flop_size)

    if tai_flop_size_ > 300:
        tai_flop_size_ = int(tai_flop_size_) # Handle overflow errors
    
    algo_doubling_rate_ = algo_halving_fn(sq.sample(algo_doubling_rate_min),
                                          sq.sample(algo_doubling_rate_max),
                                          tai_flop_size_)
    
    if use_efficiency_based_algo_reduction:
        possible_algo_reduction_ = possible_algo_reduction_fn(efficiency_, efficiency_, tai_flop_size_)
        if efficiency_based_additional_reduction:
            possible_algo_reduction_ += sq.sample(efficiency_based_additional_reduction)
    else:
        possible_algo_reduction_ = possible_algo_reduction_fn(sq.sample(min_reduction),
                                                              sq.sample(max_reduction),
                                                              tai_flop_size_)
    
    initial_flops_per_dollar_ = 10 ** sq.sample(initial_flops_per_dollar)
    flops_halving_rate_ = sq.sample(flops_halving_rate)
    max_flops_per_dollar_ = 10 ** sq.sample(max_flops_per_dollar)
    initial_pay_ = 10 ** sq.sample(initial_pay)
    gdp_growth_ = sq.sample(gdp_growth)
    max_gdp_frac_ = max_gdp_frac(state['war'])
    
    willingness_ramp_happens = sq.event_occurs(p_willingness_ramp)
    if willingness_ramp_happens:
        willingness_ramp_ = sq.sample(willingness_ramp)
    else:
        willingness_ramp_ = 1
    
    initial_gdp_ = initial_gdp
    spend_doubling_time_ = sq.sample(spend_doubling_time)
    nonscaling_delay_ = sq.sample(nonscaling_delay)
    nonscaling_countdown = 0
    initial_chance_of_nonscaling_issue_ = initial_chance_of_nonscaling_issue
    final_chance_of_nonscaling_issue_ = final_chance_of_nonscaling_issue
    nonscaling_issue_bottom_year_ = nonscaling_issue_bottom_year
    willingness_spend_horizon_ = int(sq.sample(willingness_spend_horizon))
    tai_china_war_delay_yrs_ = None
    tai_catastrophe_delay_yrs_ = None
    
    queue_tai_year = 99999
    plt.ioff()
    if verbosity > 0:
        cost_of_tai_collector = []
        willingness_collector = []
    
    if verbosity > 1:
        print('### TAI TIMELINE VARIABLES ###')
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
                                                              np.round(max_gdp_frac_, 5),
                                                              np.round(gdp_growth_, 3)))
        if willingness_ramp_ < 1:
            print('In this simulation, if we are {}% of the way to paying for TAI, we will ramp to paying for TAI.'.format(np.round(willingness_ramp_ * 100)))

        if willingness_spend_horizon_ > 1:
            print('We are willing to spend over {} years to make TAI'.format(willingness_spend_horizon_))
            
        print(('If a non-scaling delay happens, it will take an additional {} years to produce TAI due' +
               ' to issues unrelated to scaling FLOP').format(np.round(nonscaling_delay_, 1)))
        print('-')
    
    is_nonscaling_issue = None

    for y in years:
        n_catastrophes = len(state['catastrophe'])
        
        # Run modules
        # TODO: Run in random order?
        if not state['terminate']:
            state = tai_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = great_power_war_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = bio_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = nuclear_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = nano_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = supervolcano_scenarios_module(y, state, verbosity > 0)
        
        if not state['terminate']:
            state = unknown_unknown_scenarios_module(y, state, verbosity > 0)
        
        # Check for double-dip catastrophe -> x-risk
        # TODO: Move to module?
        catastrophe_this_year = len(state['catastrophe']) > n_catastrophes
        if not state['terminate'] and catastrophe_this_year:
            prior_catastrophe_within_range = n_catastrophes > 0 and y < (state['recent_catastrophe_year'] + extinction_from_double_catastrophe_range)
            if prior_catastrophe_within_range and sq.event(p_extinction_from_double_catastrophe):
                previous_catastrophe = state['catastrophe'][-2]
                current_catastrophe = state['catastrophe'][-1]
                if verbosity > 0:
                    print('{}: ...XRISK from double catastrophe ({} -> {})'.format(y, previous_catastrophe, current_catastrophe))
                
                if current_catastrophe == 'averting_intentional_tai' or current_catastrophe == 'averting_misaligned_tai':
                    xrisk_name = 'xrisk_full_unaligned_tai_extinction'
                elif current_catastrophe == 'natural_pathogen':
                    xrisk_name = 'xrisk_bio_accident'
                elif current_catastrophe == 'engineered_pathogen':
                    xrisk_name = 'xrisk_bio_accident'
                elif 'xrisk_' in current_catastrophe:
                    xrisk_name = current_catastrophe
                else:
                    xrisk_name = 'xrisk_{}'.format(current_catastrophe)
                state['category'] = xrisk_name
                state['double_catastrophe_xrisk'] = '{}->{}'.format(previous_catastrophe, current_catastrophe)
                state['terminate'] = True; state['final_year'] = y
            state['recent_catastrophe_year'] = y
            

        # Check if TAI is created this year
        # TODO: Move to module?
        if not state['terminate'] and not state['tai']:
            if catastrophe_this_year:
                tai_catastrophe_delay_yrs_ = int(round(sq.sample(tai_catastrophe_delay_yrs)))
                if nonscaling_countdown > 0:
                    nonscaling_countdown += tai_catastrophe_delay_yrs_
                    if verbosity > 1:
                        print('Catastrophe adds {} yrs to nonscaling delay'.format(tai_catastrophe_delay_yrs_))
                elif verbosity > 1:
                    print('Catastrophe delays TAI progress by {} years'.format(tai_catastrophe_delay_yrs_))
                
            if state['war'] and y == state['war_start_year']:
                old_max_gdp_frac_ = max_gdp_frac_
                max_gdp_frac_ = max_gdp_frac(war=True)
                if verbosity > 1:
                    print('Due to war, our willingness to pay for TAI has moved from {}% of GDP to {}% of GDP'.format(round(old_max_gdp_frac_, 5),
                                                                                                                      round(max_gdp_frac_, 5)))
                if state['war_belligerents'] == 'US/China' and y < tai_china_war_delay_end_year:
                    tai_china_war_delay_yrs_ = int(round(sq.sample(tai_china_war_delay_yrs)))
                    if nonscaling_countdown > 0:
                        nonscaling_countdown += tai_china_war_delay_yrs_
                        if verbosity > 1:
                            print('War with China adds {} yrs to nonscaling delay'.format(tai_china_war_delay_yrs_))
                    elif verbosity > 1:
                        print('Due to war with China, TAI progress is delayed {} years'.format(tai_china_war_delay_yrs_))
            
            if not state['war'] and y == state['war_end_year']:
                old_max_gdp_frac_ = max_gdp_frac_
                max_gdp_frac_ = max_gdp_frac(war=False)
                if verbosity > 1:
                    print('Due to war ending, our willingness to pay for TAI has moved from {}% of GDP to {}% of GDP'.format(round(old_max_gdp_frac_, 5),
                                                                                                                             round(max_gdp_frac_, 5)))
                    
            y_relative = y - CURRENT_YEAR
            tai_catastrophe_delay_yrs__ = tai_catastrophe_delay_yrs_
            if (tai_catastrophe_delay_yrs__ and
                state['recent_catastrophe_year'] and
                y_relative - tai_catastrophe_delay_yrs__ < (state['recent_catastrophe_year'] - CURRENT_YEAR)):
                    tai_catastrophe_delay_yrs__ = y - state['recent_catastrophe_year']

            tai_china_war_delay_yrs__ = tai_china_war_delay_yrs_
            if (tai_china_war_delay_yrs__ and
                state['war_start_year'] and
                y_relative - tai_china_war_delay_yrs__ < (state['war_start_year'] - CURRENT_YEAR)):
                    tai_china_war_delay_yrs__ = y - state['war_start_year']

            y_relative = y_relative - tai_catastrophe_delay_yrs__ if tai_catastrophe_delay_yrs__ else y_relative
            y_relative = y_relative - tai_china_war_delay_yrs__ if tai_china_war_delay_yrs__ else y_relative

            if verbosity > 2:
                print('- Year is {} (normal relative year is {}, adjusted relative year is {})'.format(y,
                                                                                                        y - CURRENT_YEAR,
                                                                                                        y_relative))

            flops_needed_ = flops_needed(initial_flops=10 ** tai_flop_size_,
                                         doubling_rate=algo_doubling_rate_,
                                         possible_reduction=10 ** possible_algo_reduction_,
                                         year=y_relative)

            flops_per_dollar_ = flops_per_dollar(initial_flops_per_dollar=initial_flops_per_dollar_,
                                                 max_flops_per_dollar=max_flops_per_dollar_,
                                                 halving_rate=flops_halving_rate_,
                                                 year=y_relative)

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
                                              year=y_relative)

            if flops_per_dollar_ > 10 ** 200:
                willingness_ = int(willingness_)
            if willingness_ > 10 ** 200:
                flops_per_dollar_ = int(flops_per_dollar_)

            if verbosity > 0:
                cost_of_tai_collector.append(cost_of_tai_)
                willingness_collector.append(willingness_)

            total_compute_ = willingness_ * flops_per_dollar_

            if verbosity > 2:
                out_str = (' --- {} max log FLOP ({}) available - TAI takes {} log FLOP ({}) - ' +
                           'log $ {} to buy TAI ({}) vs. willingness to pay log $ {} ({}) - {} log FLOPS per $ ({})')
                print(out_str.format(np.round(math.log10(total_compute_), 1),
                                     numerize(total_compute_),
                                     np.round(math.log10(flops_needed_), 1),
                                     numerize(flops_needed_),
                                     np.round(math.log10(cost_of_tai_), 1),
                                     numerize(cost_of_tai_),
                                     np.round(math.log10(willingness_), 1),
                                     numerize(willingness_),
                                     np.round(math.log10(flops_per_dollar_), 1),
                                     numerize(flops_per_dollar_)))
                print('-')

            if cost_of_tai_ > 10 ** 200:
                spend_tai_years = int(cost_of_tai_) // int(willingness_)
            else:
                spend_tai_years = cost_of_tai_ / willingness_

            if not is_nonscaling_issue and queue_tai_year < 99999 and verbosity > 2:
                print('-$- {}/{}'.format(y, queue_tai_year))
            if (cost_of_tai_ * willingness_ramp_) <= willingness_ or y >= queue_tai_year:
                if is_nonscaling_issue is None:
                    p_nonscaling_delay_ = p_nonscaling_delay(initial_chance_of_nonscaling_issue_,
                                                             final_chance_of_nonscaling_issue_,
                                                             year=y,
                                                             max_year=nonscaling_issue_bottom_year_)
                    is_nonscaling_issue = sq.event(p_nonscaling_delay_)
                    nonscaling_countdown = nonscaling_delay_
                    if verbosity > 2:
                        print('-- {} p_nonscaling_issue={}'.format('Nonscaling delay occured' if is_nonscaling_issue else 'Nonscaling issue did not occur',
                                                                   np.round(p_nonscaling_delay_, 4)))

                if not is_nonscaling_issue or nonscaling_countdown <= 0.1:
                    if verbosity > 0:
                        print('--- /!\ TAI CREATED in {}'.format(y))
                        plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()
                    state['tai'] = True
                    state['tai_year'] = y
                else:
                    if verbosity > 2:
                        print('{}: /!\ FLOPs for TAI sufficient but needs {} more years to solve non-scaling issues'.format(y, np.round(nonscaling_countdown, 1)))
                    nonscaling_countdown -= 1
            elif (not is_nonscaling_issue and willingness_spend_horizon_ > 1 and
                  spend_tai_years <= willingness_spend_horizon_ and y + math.ceil(spend_tai_years) < queue_tai_year):
                queue_tai_year = y + math.ceil(spend_tai_years)
                if verbosity > 2:
                    print('-$- We have enough spend to make TAI in {} years (in {}) if sustained.'.format(math.ceil(spend_tai_years),
                                                                                                          queue_tai_year))


        ## Check validity of state
        for k in list(state.keys()):
            if k not in allowed_state_keys:
                raise ValueError('key {} found and not provisioned'.format(k))
        if state['category'] not in STATES:
            raise ValueError('State {} not in `STATES`'.format(state['category']))

        ## Run collectors
        collectors[y] = deepcopy(state)
                
            
    ## Boring future if MAX_YEAR is reached with no termination
    if not state['terminate']:
        if verbosity > 0:
            print('...Boring future')
            if not state['tai']:
                plot_tai(plt, years, cost_of_tai_collector, willingness_collector).show()
        state['final_year'] = '>{}'.format(y)

    if verbosity > 0:
        print('-')
        print('-')
    
    tai_metadata = {'initial_gdp': initial_gdp_,
                    'tai_flop_size': tai_flop_size_,
                    'nonscaling_delay': nonscaling_delay_,
                    'algo_doubling_rte': algo_doubling_rate_,
                    'possible_algo_reduction': possible_algo_reduction_,
                    'initial_flops_per_dollar': initial_flops_per_dollar_,
                    'flops_halving_rate': flops_halving_rate_,
                    'max_flops_per_dollar': max_flops_per_dollar_,
                    'initial_pay': initial_pay_,
                    'gdp_growth': gdp_growth_,
                    'max_gdp_frac': max_gdp_frac_,
                    'willingness_ramp_': willingness_ramp_,
                    'spend_doubling_time_': spend_doubling_time_,
                    'initial_chance_of_nonscaling_issue': initial_chance_of_nonscaling_issue_,
                    'final_chance_of_nonscaling_issue': final_chance_of_nonscaling_issue_,
                    'nonscaling_issue_bottom_year': nonscaling_issue_bottom_year_,
                    'willingness_spend_horizon_': willingness_spend_horizon_,
                    'tai_china_war_delay_yrs': tai_china_war_delay_yrs_}
        
    return {'collectors': collectors,
            'tai_metadata': tai_metadata,
            'final_state': state}
