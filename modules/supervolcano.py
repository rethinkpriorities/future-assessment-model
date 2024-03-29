def supervolcano_scenarios_module(y, state, variables, verbose):
    if sq.event(variables['p_supervolcano_catastrophe']):
        if sq.event(variables['p_supervolcano_extinction_given_catastrophe']):
            if verbose:
                print('{}: ...XRISK from supervolcano'.format(y))
            state['category'] = 'xrisk_supervolcano'
            state['catastrophe'].append({'catastrophe': state['category'],
                                         'year': y})
            state['terminate'] = True
            state['final_year'] = y
        else:
            if verbose:
                print('{}: ...Catastrophe from supervolcano'.format(y))
            state['catastrophe'].append({'catastrophe': 'supervolcano',
                                         'year': y})
            
    return state
