def check_for_double_dip_catastrophe(y, state, variables, verbose):
    if not state['terminate'] and variables['catastrophe_this_year'] and len(state['catastrophe']) > 1:
        prior_catastrophe_within_range = y < (state['catastrophe'][-2]['year'] + variables['extinction_from_double_catastrophe_range'])
        if prior_catastrophe_within_range and sq.event(variables['p_extinction_from_double_catastrophe']):
            previous_catastrophe = state['catastrophe'][-2]
            current_catastrophe = state['catastrophe'][-1]
            if verbose:
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
            state['terminate'] = True
            state['final_year'] = y
    return state
