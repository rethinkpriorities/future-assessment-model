# TODO: Refactor
def tai_scenarios_module(y, state, verbose):
    if state['tai_type'] != 'abandoned' and state['tai_type'] != 'tool':
        # TODO: Slow vs. fast takeoff
        # TODO: May not deploy TAI as soon as it is deployable
        if (sq.event_occurs(p_make_agent_tai) or state['tai_type'] == 'agent') and state['tai_type'] != 'tool':
            # TODO: Do we want to re-roll the possibility of making agentic TAI in future years?
            if sq.event_occurs(p_tai_intentional_misuse(state['war'])):
                # TODO: Parameterize the 0.5 (Who gets to TAI first affecting risk)
                # TODO: May depend on who the war is between?
                if 0.5 * sq.event_occurs(p_alignment_solved(state['war'], y - CURRENT_YEAR, first_attempt=not state['tai'], verbose=verbose)):
                    if verbose:
                        # TODO: aligned by default? subtle misalignment?
                        # TODO: announce intentional misuse
                        print('{}: ...Achieved aligned TAI (aligned via work, {} attempt), happy future! :D'.format(y, '2nd+' if state['tai'] else 'first'))
                    state['category'] = 'aligned_tai'
                    state['tai_type'] = 'agent'
                    state['terminate'] = True; state['final_year'] = y
                elif sq.event_occurs(p_full_tai_misalignment_averted):
                    state['averted_misalignment'] = True
                    if sq.event_occurs(p_tai_misalignment_averting_is_catastrophic):
                        # TODO: Right now this guarantees abandonment if catastrophe - revise?
                        if verbose:
                            print('{}: ...Intentional misuse of TAI happened, was averted with catastrophe, and we abandon TAI'.format(y))
                        state['tai_type'] = 'abandoned'
                        state['catastrophe'].append('averting_intentional_tai')
                        # TODO: Maybe resume TAI with lower chance of happening?
                    elif sq.event_occurs(p_full_tai_misalignment_averted_means_abandoned_tai):
                        if verbose:
                            print('{}: ...Intentional misuse of TAI happened, it was averted with no catastrophe, and we abandon TAI'.format(y))
                        state['tai_type'] = 'abandoned'    
                        # TODO: Maybe resume TAI with lower chance of happening?
                    elif verbose:
                        print('{}: ...Intentional misuse of TAI happened, but it was averted'.format(y))
                else:
                    if verbose:
                        print('{}: ...XRISK from intentional misuse of TAI (singleton) :('.format(y))
                    state['category'] = 'xrisk_tai_misuse'
                    state['tai_type'] = 'agent'
                    if sq.event_occurs(p_tai_singleton_is_catastrophic):
                        if verbose:
                            print('...Singleton is catastrophic')
                        state['catastrophe'].append(state['category'])
                    state['terminate'] = True; state['final_year'] = y
            elif sq.event_occurs(p_alignment_solved(state['war'], y - CURRENT_YEAR, first_attempt=not state['tai'], verbose=verbose)):
                if sq.event_occurs(p_subtle_alignment_solved):
                    if verbose:
                        print('{}: ...Achieved aligned TAI (aligned via work, {} attempt), happy future! :D'.format(y, '2nd+' if state['tai'] else 'first'))
                    state['category'] = 'aligned_tai'
                    state['tai_type'] = 'agent'
                    state['terminate'] = True; state['final_year'] = y
                else:
                    if verbose:
                        print('{}: ...XRISK from subtly unaligned TAI :('.format(y))
                    state['category'] = 'xrisk_subtly_unaligned_tai'
                    state['tai_type'] = 'agent'
                    state['terminate'] = True; state['final_year'] = y
            elif sq.event_occurs(p_tai_aligned_by_default):
                if sq.event_occurs(p_subtle_alignment_solved_if_aligned_by_default):
                    if verbose:
                        print('{}: ...Achieved aligned TAI (aligned by default), happy future! :D'.format(y))
                    state['category'] = 'aligned_tai'
                    state['tai_type'] = 'agent'
                    state['terminate'] = True; state['final_year'] = y
                    # TODO: Does aligned TAI make all other xrisks impossible? We currently assume it does
                        # TODO: Maybe introduce takeoff delay before other xrisks become impossible?
                else:
                    if verbose:
                        print('{}: ...XRISK from subtly unaligned TAI :('.format(y))
                    state['category'] = 'xrisk_subtly_unaligned_tai'
                    state['tai_type'] = 'agent'
                    state['terminate'] = True; state['final_year'] = y
            elif sq.event_occurs(p_full_tai_misalignment_averted):
                state['averted_misalignment'] = True
                if sq.event_occurs(p_tai_misalignment_averting_is_catastrophic):
                    # TODO: Right now this guarantees abandonment if catastrophe - revise?
                    if verbose:
                        print('{}: ...Misaligned TAI happened, it was averted with catastrophe, and we abandon TAI'.format(y))
                    state['tai_type'] = 'abandoned'
                    state['catastrophe'].append('averting_misaligned_tai')
                    # TODO: Maybe resume TAI with lower chance of happening?
                elif sq.event_occurs(p_full_tai_misalignment_averted_means_abandoned_tai):
                    if verbose:
                        print('{}: ...Misaligned TAI happened, it was averted with no catastrophe, and we abandon TAI'.format(y))
                    state['tai_type'] = 'abandoned'    
                    # TODO: Maybe resume TAI with lower chance of happening?
                elif verbose:
                    print('{}: ...Misaligned TAI happened, but it was averted'.format(y))
                    state['tai_type'] = 'agent'
            elif sq.event_occurs(p_tai_xrisk_is_extinction):
                if verbose:
                    print('{}: ...XRISK from fully unaligned TAI (extinction) :('.format(y))
                state['category'] = 'xrisk_full_unaligned_tai_extinction'
                state['tai_type'] = 'agent'
                state['catastrophe'].append(state['category'])
                state['terminate'] = True; state['final_year'] = y
            else:
                if verbose:
                    print('{}: ...XRISK from fully unaligned TAI (singleton) :('.format(y))
                state['category'] = 'xrisk_full_unaligned_tai_singleton'
                state['tai_type'] = 'agent'
                if sq.event_occurs(p_tai_singleton_is_catastrophic):
                    if verbose:
                        print('...Singleton is catastrophic')
                    state['catastrophe'].append(state['category'])
                state['terminate'] = True; state['final_year'] = y
        else: # tool TAI
            if verbose:
                print('{}: ...Tool TAI made'.format(y))
            state['tai_type'] = 'tool'
            # TODO: Does tool AI help mitigate xrisk?
    return state
# TODO: TAI or nearness to TAI creates great power war
