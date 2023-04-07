def bio_scenarios_module(y, state, variables, verbose):
    if sq.event(variables['p_natural_bio'](y - variables['CURRENT_YEAR'], variables)):
        state['natural_pathogen'] = True
        if sq.event(variables['p_natural_bio_is_catastrophe']):
            if sq.event(variables['p_xrisk_from_accidental_bio_given_catastrophe'](y - variables['CURRENT_YEAR'], variables)):
                if verbose:
                    print('{}: ...XRISK from pathogen (lab-leak) :('.format(y))
                state['category'] = 'xrisk_bio_accident'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('natural_pathogen')
            else:
                if verbose:
                    print('{}: ...catastrophe from natural pathogen'.format(y))
                state['catastrophe'].append('natural_pathogen')

    if not state['terminate'] and sq.event(variables['p_accidental_bio'](state['war'], variables)):
        state['lab_leak'] = True
        engineered = sq.event(variables['ratio_engineered_vs_natural_lab_leak'])
        if engineered:
            state['engineered_pathogen'] = True
        else:
            state['natural_pathogen'] = True
        if sq.event(variables['p_engineered_bio_is_catastrophe']):
            if sq.event(variables['p_xrisk_from_accidental_bio_given_catastrophe'](y - variables['CURRENT_YEAR'], variables)):
                if verbose:
                    print('{}: ...XRISK from pathogen (lab-leak) :('.format(y))
                state['category'] = 'xrisk_bio_accident'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('engineered_pathogen' if engineered else 'natural_pathogen')
            else:
                if verbose:
                    if engineered:
                        print('{}: ...catastrophe from lab-leak engineered pathogen'.format(y))
                    else:
                        print('{}: ...catastrophe from lab-leak natural pathogen'.format(y))
                state['catastrophe'].append('engineered_pathogen' if engineered else 'natural_pathogen')
    
    if not state['terminate'] and state['war'] and sq.event(variables['p_biowar_given_war']):
        state['engineered_pathogen'] = True
        state['state_bioweapon'] = True
        if sq.event(variables['p_engineered_bio_is_catastrophe']):
            if sq.event(variables['p_xrisk_from_engineered_bio_given_catastrophe'](y - variables['CURRENT_YEAR'], variables)):
                if verbose:
                    print('{}: ...XRISK from pathogen (war) :('.format(y))
                state['category'] = 'xrisk_bio_war'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('engineered_pathogen')
            else:
                if verbose:
                    print('{}: ...catastrophe from pathogen (war)'.format(y))
                state['catastrophe'].append('engineered_pathogen')
    
    if not state['terminate'] and sq.event(variables['p_nonstate_bio']):
        state['nonstate_bioweapon'] = True
        engineered = sq.event(variables['ratio_engineered_vs_natural_lab_leak'])
        if engineered:
            state['engineered_pathogen'] = True
        else:
            state['natural_pathogen'] = True
        if sq.event(variables['p_engineered_bio_is_catastrophe']):
            if sq.event(variables['p_xrisk_from_engineered_bio_given_catastrophe'](y - variables['CURRENT_YEAR'], variables)):
                if verbose:
                    print('{}: ...XRISK from pathogen (nonstate) :('.format(y))
                state['category'] = 'xrisk_bio_nonstate'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('engineered_pathogen' if engineered else 'natural_pathogen')
            else:
                if verbose:
                    if engineered:
                        print('{}: ...catastrophe from engineered pathogen'.format(y))
                    else:
                        print('{}: ...catastrophe from natural pathogen'.format(y))
                state['catastrophe'].append('engineered_pathogen' if engineered else 'natural_pathogen')
                
    return state
