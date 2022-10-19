def supervolcano_scenarios_module(y, state, verbose):
    if sq.event_occurs(p_supervolcano_catastrophe):
        if sq.event_occurs(p_supervolcano_extinction_given_catastrophe):
            if verbose:
                print('{}: ...XRISK from supervolcano'.format(y))
            state['category'] = 'xrisk_supervolcano'
            state['catastrophe'].append(state['category'])
            state['terminate'] = True; state['final_year'] = y
        else:
            if verbose:
                print('{}: ...Catastrophe from supervolcano'.format(y))
            state['catastrophe'].append('supervolcano')
            
    return state
