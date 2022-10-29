def great_power_war_scenarios_module(y, state, verbose):
    peace = y < state['peace_until'] if state['peace_until'] is not None else False
    
    if not state['war'] and (sq.event(p_great_power_war_us_russia_without_nuke_first(peace, y - CURRENT_YEAR)) or
                             state['russia_nuke_first']):
        if verbose:
            print('{}: WAR!!! (US vs. Russia)'.format(y))
        state['war'] = True
        state['war_start_year'] = y
        war_length_ = int(round(~war_length))
        state['war_end_year'] = war_length_ + y
        state['war_belligerents'] = 'US/Russia'
        state['wars'].append({'belligerents': state['war_belligerents'],
                              'start_year': state['war_start_year'],
                              'end_year': state['war_end_year'],
                              'war_length': war_length_})
        state['russia_nuke_first'] = False
        
    elif not state['war'] and (sq.event(p_great_power_war_us_china(peace, y - CURRENT_YEAR)) or
                               state['china_nuke_first']):
        if verbose:
            print('{}: WAR!!! (US vs. China)'.format(y))
        state['war'] = True
        state['war_start_year'] = y
        war_length_ = int(round(~war_length))
        state['war_end_year'] = war_length_ + y
        state['war_belligerents'] = 'US/China'
        state['wars'].append({'belligerents': state['war_belligerents'],
                              'start_year': state['war_start_year'],
                              'end_year': state['war_end_year'],
                              'war_length': war_length_})
    
    elif sq.event(p_great_power_war_other(peace, y - CURRENT_YEAR)) and not state['war']:
        if verbose:
            print('{}: WAR!!! (Other)'.format(y))
        state['war'] = True
        state['war_start_year'] = y
        war_length_ = int(round(~war_length))
        state['war_end_year'] = war_length_ + y
        state['war_belligerents'] = 'Other'
        state['wars'].append({'belligerents': state['war_belligerents'],
                              'start_year': state['war_start_year'],
                              'end_year': state['war_end_year'],
                              'war_length': war_length_})
        
    elif state['war'] and (y >= state['war_end_year'] or state['category'] == 'aligned_tai'):
        if verbose:
            print('{}: War ends :)'.format(y))
        state['war'] = False
        state['war_belligerents'] = None
        peace_length_ = int(round(~peace_length))
        state['peace_until'] = y + peace_length_
        
    return state

# TODO: There should be a small chance of catastrophe and x-risk from great power war, absent nuclear or bio pathway
# TODO: There should be a chance of nukes-related catastrophe and singleton x-risk from breaking nuclear symmetry
