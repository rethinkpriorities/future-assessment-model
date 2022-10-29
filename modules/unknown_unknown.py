def unknown_unknown_scenarios_module(y, state, verbose):
    if sq.event(p_unknown_unknown_xrisk(y - CURRENT_YEAR)):
        if verbose:
            print('{}: ...XRISK from unknown unknown'.format(y))
        state['category'] = 'xrisk_unknown_unknown'
        state['catastrophe'].append(state['category'])
        # TODO: What % of "unknown unknowns" are extinction?
        # TODO: catastrophic but not x-risk unknown unknowns?
        state['terminate'] = True; state['final_year'] = y
        
    return state
