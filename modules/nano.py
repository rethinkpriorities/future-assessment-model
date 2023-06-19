def nano_scenarios_module(y, state, variables, verbose):
    if sq.event(variables['p_nanotech_possible'](y - variables['CURRENT_YEAR'])) and not state['nano']:
        state['nano'] = True
        if sq.event(variables['p_nanotech_is_xrisk']):
            if verbose:
                print('{}: ...XRISK from nanotech :('.format(y))
            state['category'] = 'xrisk_nanotech'
            state['catastrophe'].append({'catastrophe': state['category'], 'year': y})
            state['terminate'] = True
            state['final_year'] = y
        # TODO: Catastrophic but non x-risk nano?
        # TODO: catastrophe or nano x-risk but not in the first year?
                        
    return state
