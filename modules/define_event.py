import time


STATES = ['boring', 'xrisk_tai_misuse', 'aligned_tai', 'xrisk_full_unaligned_tai_extinction',
          'xrisk_full_unaligned_tai_singleton', 'xrisk_subtly_unaligned_tai',
          'xrisk_unknown_unknown', 'xrisk_nanotech', 'xrisk_nukes_war', 'xrisk_nukes_accident',
          'xrisk_bio_accident', 'xrisk_bio_war', 'xrisk_bio_nonstate', 'xrisk_supervolcano']

extinctions = ['xrisk_full_unaligned_tai_extinction', 'xrisk_nukes_war', 'xrisk_nukes_accident',
               'xrisk_unknown_unknown', 'xrisk_nanotech', 'xrisk_bio_accident', 'xrisk_bio_war',
               'xrisk_bio_nonstate', 'xrisk_supervolcano']


def define_event(verbosity=0):
    if verbosity is True:
        verbosity = 1

    if verbosity > 1:
        start1 = time.time()

    state = {'category': 'boring', 'tai': False, 'tai_year': None, 'tai_type': None,
             'nano': False, 'wars': [], 'war': False, 'war_start_year': None,
             'war_end_year': None, 'russia_nuke_first': False, 'china_nuke_first': False,
             'war_belligerents': None, 'peace_until': None, 'engineered_pathogen': False,
             'natural_pathogen': False, 'lab_leak': False, 'state_bioweapon': False,
             'nonstate_bioweapon': False, 'averted_misalignment': False,
             'nuclear_weapon_used': False, 'catastrophe': [], 'recent_catastrophe_year': None,
             'terminate': False, 'final_year': None, 'double_catastrophe_xrisk': None}
    allowed_state_keys = list(state.keys())
    collectors = {}
    tai_year = sq.sample(sq.discrete(tai_years))
    
    for y in years:
        n_catastrophes = len(state['catastrophe'])
        
        # Run modules
        # TODO: Run in random order?
        if not state['terminate'] and state['tai']:
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
            if state['recent_catastrophe_year'] is None:
                state['recent_catastrophe_year'] = y
            else:
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
                else:
                    state['recent_catastrophe_year'] = y
            

        # Check if TAI is created this year
        if not state['terminate'] and not state['tai'] and y >= tai_year:
            if verbosity > 0:
                print('--- /!\ TAI CREATED in {}'.format(y))
            state['tai'] = True
            state['tai_year'] = y


        ## Check validity of state
        for k in list(state.keys()):
            if k not in allowed_state_keys:
                raise ValueError('key {} found and not provisioned'.format(k))
        if state['category'] not in STATES:
            raise ValueError('State {} not in `STATES`'.format(state['category']))

        ## Run collectors
        collectors[y] = deepcopy(state)

    if verbosity > 1:
        _mark_time(start1, label='Total loop complete')
                
            
    ## Boring future if MAX_YEAR is reached with no termination
    if not state['terminate']:
        if verbosity > 0:
            print('...Boring future')
        state['final_year'] = '>{}'.format(y)

    if verbosity > 0:
        print('-')
        print('-')
    
    return {'collectors': collectors, 'final_state': state}
