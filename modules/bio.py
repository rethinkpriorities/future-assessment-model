def bio_scenarios_module(y, state, verbose):
    if sq.event_occurs(p_natural_bio(y - CURRENT_YEAR)):
        state['natural_pathogen'] = True
        if sq.event_occurs(p_natural_bio_is_catastrophe):
            if sq.event_occurs(p_xrisk_from_accidental_bio_given_catastrophe(y - CURRENT_YEAR)):
                if verbose:
                    print('{}: ...XRISK from pathogen (lab-leak) :('.format(y))
                state['category'] = 'xrisk_bio_accident'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('natural_pathogen')
            else:
                if verbose:
                    print('{}: ...catastrophe from natural pathogen'.format(y))
                state['catastrophe'].append('natural_pathogen')

    if not state['terminate'] and sq.event_occurs(p_accidental_bio(state['war'])):
        engineered = sq.event_occurs(ratio_engineered_vs_natural_lab_leak)
        if engineered:
            state['engineered_pathogen'] = True
        else:
            state['natural_pathogen'] = True
        if sq.event_occurs(p_engineered_bio_is_catastrophe):
            if sq.event_occurs(p_xrisk_from_accidental_bio_given_catastrophe(y - CURRENT_YEAR)):
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
    
    if not state['terminate'] and state['war'] and sq.event_occurs(p_biowar_given_war):
        state['engineered_pathogen'] = True
        if sq.event_occurs(p_engineered_bio_is_catastrophe):
            if sq.event_occurs(p_xrisk_from_engineered_bio_given_catastrophe(y - CURRENT_YEAR)):
                if verbose:
                    print('{}: ...XRISK from pathogen (war) :('.format(y))
                state['category'] = 'xrisk_bio_war'
                state['terminate'] = True; state['final_year'] = y
                state['catastrophe'].append('engineered_pathogen')
            else:
                if verbose:
                    print('{}: ...catastrophe from pathogen (war)'.format(y))
                state['catastrophe'].append('engineered_pathogen')
    
    if not state['terminate'] and sq.event_occurs(p_nonstate_bio):
        engineered = sq.event_occurs(ratio_engineered_vs_natural_lab_leak)
        if engineered:
            state['engineered_pathogen'] = True
        else:
            state['natural_pathogen'] = True
        if sq.event_occurs(p_engineered_bio_is_catastrophe):
            if sq.event_occurs(p_xrisk_from_engineered_bio_given_catastrophe(y - CURRENT_YEAR)):
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
