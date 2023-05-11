import random
import time


STATES = ['boring', 'xrisk_tai_misuse', 'aligned_tai', 'xrisk_full_unaligned_tai_extinction',
          'xrisk_full_unaligned_tai_singleton', 'xrisk_subtly_unaligned_tai',
          'xrisk_unknown_unknown', 'xrisk_nanotech', 'xrisk_nukes_war', 'xrisk_nukes_accident',
          'xrisk_bio_accident', 'xrisk_bio_war', 'xrisk_bio_nonstate', 'xrisk_supervolcano']

extinctions = ['xrisk_full_unaligned_tai_extinction', 'xrisk_nukes_war', 'xrisk_nukes_accident',
               'xrisk_unknown_unknown', 'xrisk_nanotech', 'xrisk_bio_accident', 'xrisk_bio_war',
               'xrisk_bio_nonstate', 'xrisk_supervolcano']


def define_event(variables, verbosity=0):
    if verbosity is True:
        verbosity = 1

    if verbosity > 1:
        start1 = time.time()

    state = {'category': 'boring', 'tai': False, 'tai_year': None, 'tai_type': None,
             'tai_alignment_state': None, 'nano': False, 'wars': [], 'war': False,
             'war_start_year': None, 'war_end_year': None, 'russia_nuke_first': False,
             'china_nuke_first': False, 'war_belligerents': None, 'peace_until': None,
             'engineered_pathogen': False, 'natural_pathogen': False, 'lab_leak': False,
             'state_bioweapon': False, 'nonstate_bioweapon': False, 'averted_misalignment': False,
             'nuclear_weapon_used': False, 'catastrophe': [], 'recent_catastrophe_year': None,
             'terminate': False, 'final_year': None, 'double_catastrophe_xrisk': None}
    allowed_state_keys = list(state.keys())
    collectors = {}
    tai_year = sq.sample(sq.discrete([y['tai_year'] for y in variables['tai_years']]))
    us_china_war_tai_delay_has_occurred = False
    
    for y in years:
        n_catastrophes = len(state['catastrophe'])

        # Run modules in a random order
        modules = [tai_scenarios_module,
                   great_power_war_scenarios_module,
                   bio_scenarios_module,
                   nuclear_scenarios_module,
                   nano_scenarios_module,
                   supervolcano_scenarios_module,
                   unknown_unknown_scenarios_module]
        random.shuffle(modules)
        for module in modules:
            if not state['terminate']:
                state = module(y, state, variables, verbosity)
        
        # Check for double dip catastrophe
        catastrophe_this_year = len(state['catastrophe']) > n_catastrophes
        state = check_for_double_dip_catastrophe(y,
                                                 state,
                                                 variables,
                                                 catastrophe_this_year,
                                                 n_catastrophes,
                                                 verbosity)

        # Modify TAI arrival date for wars and catastrophes
        if not state['terminate'] and not state['tai']:
            if catastrophe_this_year:
                delay = int(np.ceil(sq.sample(variables['if_catastrophe_delay_tai_arrival_by_years'])))
                tai_year += delay
                if verbosity > 0:
                    print('...catastrophe delays TAI by {} years'.format(delay))
            if (state['war'] and
                state['war_start_year'] == y and
                not us_china_war_tai_delay_has_occurred and
                state['war_belligerents'] == 'US/China'):
                us_china_war_tai_delay_has_occurred = True
                delay = int(np.ceil(sq.sample(variables['if_us_china_war_delay_tai_arrival_by_years'])))
                tai_year += delay
                if verbosity > 0:
                    print('...US-China war delays TAI by {} years'.format(delay))
        # TODO: War -> TAI spending increase

        # Check if TAI is created this year
        if not state['terminate'] and not state['tai'] and y >= tai_year:
            if verbosity > 0:
                print('--- /!\ TAI CREATED in {}'.format(y))
            state['tai'] = True
            state['tai_year'] = y

        # Enforce validity of state
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
