def nuclear_scenarios_module(y, state, variables, verbose):
    peace = y < state['peace_until'] if state['peace_until'] is not None else False
    if sq.event(variables['p_nuclear_accident'](state['war'], y - variables['CURRENT_YEAR'])):
        if sq.event(variables['p_nuclear_accident_becomes_exchange'](state['war'])):
            state['nuclear_weapon_used'].append(y)
            if sq.event(variables['p_catastrophe_from_nuclear_exchange'](state['war'])):
                if sq.event(variables['p_xrisk_from_nuclear_catastrophe']):
                    if verbose:
                        print('{}: ...XRISK from nukes (accidental exchange) :('.format(y))
                    state['category'] = 'xrisk_nukes_accident'
                    state['terminate'] = True
                    state['final_year'] = y
                else:    
                    if verbose:
                        print('{}: ...catastrophe from nukes (accidental exchange)'.format(y))
                    state['catastrophe'].append({'catastrophe': 'nukes_accident', 'year': y})
    
    first_year_of_war = state['war'] and (state['war_start_year'] == y)
    if not state['terminate'] and state['war'] and sq.event(variables['p_nuclear_exchange_given_war'](first_year_of_war)):
        state['nuclear_weapon_used'].append(y)
        if sq.event(variables['p_catastrophe_from_nuclear_exchange'](state['war'])):
            if sq.event(variables['p_xrisk_from_nuclear_catastrophe']):
                if verbose:
                    print('{}: ...XRISK from nukes (war) :('.format(y))
                state['category'] = 'xrisk_nukes_war'
                state['terminate'] = True
                state['final_year'] = y
            else:    
                if verbose:
                    print('{}: ...catastrophe from nukes (war)'.format(y))
                state['catastrophe'].append({'catastrophe': 'nukes_war', 'year': y})
    
    if not state['terminate'] and not state['war'] and sq.event(variables['p_russia_uses_nuke'](peace, y - variables['CURRENT_YEAR'], variables)):
        if verbose:
            print('{}: Russia uses a nuke first strike (outside of great power war)!'.format(y))
        state['nuclear_weapon_used'].append(y)
        state['russia_nuke_first'] = True
        
    if not state['terminate'] and sq.event(variables['p_nk_uses_nuke']):
        if verbose:
            print('{}: North Korea uses a nuke first strike!'.format(y))
        state['nuclear_weapon_used'].append(y)

    if not state['terminate'] and sq.event(variables['p_china_uses_nuke'](peace, y, variables)):
        if verbose:
            print('{}: China uses a nuke first strike (outside of a great power war)!'.format(y))
        state['nuclear_weapon_used'].append(y)
        state['china_nuke_first'] = True
    
    if not state['terminate'] and not state['war'] and sq.event(variables['p_other_uses_nuke'](peace)):
        if verbose:
            print('{}: A country other than Russia/China/NK uses a nuke first strike (outside of great power war)!'.format(y))
        state['nuclear_weapon_used'].append(y)
                
    return state
