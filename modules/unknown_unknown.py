def unknown_unknown_scenarios_module(y, state, variables, verbose):
    if sq.event(variables['p_unknown_unknown_xrisk'](y - variables['CURRENT_YEAR'])):
        if verbose:
            print('{}: ...XRISK from unknown unknown'.format(y))
        state['category'] = 'xrisk_unknown_unknown'
        state['catastrophe'].append({'catastrophe': state['category'],
                                     'year': y})
        # TODO: What % of "unknown unknowns" are extinction?
        # TODO: catastrophic but not x-risk unknown unknowns?
        state['terminate'] = True
        state['final_year'] = y
        
    return state
