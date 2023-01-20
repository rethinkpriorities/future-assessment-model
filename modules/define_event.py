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
        
        catastrophe_this_year = len(state['catastrophe']) > n_catastrophes
        state = check_for_double_dip_catastrophe(y, state, catastrophe_this_year, verbosity > 0)

        # Check if TAI is created this year
        if not state['terminate'] and not state['tai'] and y >= tai_year:
            if verbosity > 0:
                print('--- /!\ TAI CREATED in {}'.format(y))
            state['tai'] = True
            state['tai_year'] = y

        # Check validity of state
        for k in list(state.keys()):
            if k not in allowed_state_keys:
                raise ValueError('key {} found and not provisioned'.format(k))
        if state['category'] not in STATES:
            raise ValueError('State {} not in `STATES`'.format(state['category']))

        # Run collectors
        collectors[y] = deepcopy(state)

    if verbosity > 1:
        _mark_time(start1, label='Total loop complete')
            
    # Boring future if MAX_YEAR is reached with no termination
    if not state['terminate']:
        if verbosity > 0:
            print('...Boring future')
        state['final_year'] = '>{}'.format(y)

    if verbosity > 0:
        print('-')
        print('-')
    
    return {'collectors': collectors, 'final_state': state}
