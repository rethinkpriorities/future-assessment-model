import numpy as np


def deliberate_misuse(state, variables, verbosity):
    return p_event(variables['p_tai_intentional_misuse'](state['war']), 'p_tai_intentional_misuse', verbosity)


def attempt_to_avert_misaligned_tai(state, variables, y, verbosity, intentional_misuse):
    msg = 'Intentional misuse of TAI' if intentional_misuse else 'AI takeover'
    if p_event(variables, 'p_full_tai_misalignment_averted', verbosity):
        state['averted_misalignment'] = True
        if p_event(variables, 'p_tai_misalignment_averting_is_catastrophic', verbosity): # TODO: Right now this guarantees abandonment if catastrophe - revise?
            if verbosity:
                print('{}: ...{} happened, was averted with catastrophe, and we abandon TAI'.format(y, msg))
            state['tai_type'] = 'abandoned' # TODO: Maybe resume TAI with lower chance of happening?
            if intentional_misuse:
                state['tai_alignment_state'] = 'deliberate_misuse'
                state['catastrophe'].append({'catastrophe': 'averting_intentional_tai',
                                             'year': y})
            else:
                state['catastrophe'].append({'catastrophe': 'averting_misaligned_tai',
                                             'year': y})
        elif p_event(variables, 'p_full_tai_misalignment_averted_means_abandoned_tai', verbosity):
            if verbosity:
                print('{}: ...{} happened, it was averted with no catastrophe, and we abandon TAI'.format(y, msg))
            state['tai_type'] = 'abandoned' # TODO: Maybe resume TAI with lower chance of happening?
            if intentional_misuse:
                state['tai_alignment_state'] = 'deliberate_misuse'
        elif verbosity:
            print('{}: ...{} happened, but it was averted with no catastrophe'.format(y, msg))
    elif not intentional_misuse and p_event(variables, 'p_tai_xrisk_is_extinction', verbosity):
        if verbosity:
            print('{}: ...XRISK from fully unaligned TAI (extinction) :('.format(y))
        state['category'] = 'xrisk_full_unaligned_tai_extinction'
        state['tai_type'] = 'agent'
        state['catastrophe'].append({'catastrophe': state['category'],
                                     'year': y})
        state['terminate'] = True
        state['final_year'] = y
        state['tai'] = True
        state['tai_year'] = y
    elif intentional_misuse and p_event(variables, 'p_intentional_tai_singleton_creates_extinction', verbosity):
        if verbosity:
            print('{}: ...XRISK from TAI intentional misuse - attempt at singleton causes extinction :('.format(y))
        state['category'] = 'xrisk_tai_misuse_extinction'
        state['tai_type'] = 'agent'
        state['tai_alignment_state'] = 'deliberate_misuse'
        state['terminate'] = True
        state['final_year'] = y
        state['tai'] = True
        state['tai_year'] = y
    else:
        if verbosity:
            print('{}: ...XRISK from {} (singleton) :('.format(y, msg))
        if intentional_misuse:
            state['category'] = 'xrisk_tai_misuse'
            state['tai_alignment_state'] = 'deliberate_misuse'
        else:
            state['category'] = 'xrisk_full_unaligned_tai_singleton'
        state['tai_type'] = 'agent'
        if p_event(variables, 'p_tai_singleton_is_catastrophic', verbosity):
            if verbosity:
                print('...Singleton is catastrophic')
        state['catastrophe'].append({'catastrophe': state['category'],
                                     'year': y})
        state['terminate'] = True
        state['final_year'] = y
        state['tai'] = True
        state['tai_year'] = y

    return state


def tai_alignment(y, state, variables, verbosity):
    # TODO: Alignment chance should also depend a bit on who gets there first?
    aligned_by_default = p_event(variables, 'p_tai_aligned_by_default', verbosity)
    fully_aligned_by_default = aligned_by_default and p_event(variables, 'p_subtle_alignment_solved_if_aligned_by_default', verbosity)

    solved_alignment = p_event(variables['p_alignment_solved'](y - variables['CURRENT_YEAR'],
                                                               first_attempt=state['tai_type'] is None),
                               'p_alignment_solved',
                               verbosity)
    solved_subtle_alignment = p_event(variables, 'p_subtle_alignment_solved', verbosity) if solved_alignment else False

    if aligned_by_default and fully_aligned_by_default:
        state['tai_alignment_state'] = 'fully_aligned_by_default'
    elif solved_alignment and solved_subtle_alignment:
        state['tai_alignment_state'] = 'fully_aligned_by_work'
    elif aligned_by_default and not fully_aligned_by_default and solved_alignment and solved_subtle_alignment:
        state['tai_alignment_state'] = 'fully_aligned_by_work'
    elif solved_alignment and not solved_subtle_alignment:
        state['tai_alignment_state'] = 'subtly_misaligned'
    elif not solved_alignment:
        state['tai_alignment_state'] = 'blatantly_misaligned'

    return state


def coordinate_against_deployment(y, state, variables, verbosity):
    return p_event(variables['p_alignment_deploy_coordination'](state['war'],
                                                                y - variables['CURRENT_YEAR'],
                                                                variables,
                                                                first_attempt=state['tai_type'] is None),
                   'p_alignment_deploy_coordination',
                   verbosity)


def deploy_tai(y, state, variables, verbosity):
    # TODO: Update
    flop_ = np.log10(variables['effective_flop'])
    pasta = variables['threat_model'][1][1]

    if state['tai_alignment_state'] == 'fully_aligned_by_default':
        if flop_ > pasta:
            if verbosity:
                print('{}: ...Achieved aligned TAI (aligned by default)'.format(y))
            state['tai_type'] = 'aligned_agent'
            state['tai'] = True
            state['tai_year'] = y

    elif state['tai_alignment_state'] == 'fully_aligned_by_work':
        if flop_ > pasta:
            if verbosity:
                print('{}: ...Achieved aligned TAI (aligned via work, {} attempt)'.format(y, 'first' if state['tai_type'] is None else '2nd+'))
            state['tai_type'] = 'aligned_agent'
            state['tai'] = True
            state['tai_year'] = y

    elif state['tai_alignment_state'] == 'subtly_misaligned':
        if flop_ > pasta:
            if verbosity:
                print('{}: ...XRISK from subtly unaligned TAI :('.format(y))
            state['category'] = 'xrisk_subtly_unaligned_tai'
            state['tai_type'] = 'agent'
            state['terminate'] = True
            state['final_year'] = y
            state['tai'] = True
            state['tai_year'] = y

    elif state['tai_alignment_state'] == 'blatantly_misaligned':
        state = attempt_to_avert_misaligned_tai(state, variables, y, verbosity, intentional_misuse=False)

    return state


# TODO: Slow vs. fast takeoff
# TODO: May not deploy TAI as soon as it is deployable
# TODO: Do we want to re-roll the possibility of making agentic TAI in future years?
# TODO: TAI or nearness to TAI creates great power war
# TODO: Catastrophes from AI
def tai_scenarios_module(y, state, variables, verbosity):
    if state['tai_type'] != 'abandoned' and state['tai_type'] != 'aligned_agent':
        flop_ = np.log10(variables['effective_flop'])

        # TODO: Find some way to not hardcore these
        # TODO: Incorporate delays
        narrower_threat = variables['threat_model'][0][1]
        pasta = variables['threat_model'][1][1]
        
        is_deliberate_misuse_ = deliberate_misuse(state, variables, verbosity)

        if verbosity > 1:
            print('{}: Effective FLOP of {} vs. narrower threat anchor of {} and PASTA anchor of {}'.format(y,
                                                                                                            round(flop_, 1),
                                                                                                            round(narrower_threat, 1),
                                                                                                            round(pasta, 1)))

        narrower_threat_via_misuse_met = flop_ >= narrower_threat and is_deliberate_misuse_ and variables['narrower_threat_is_xrisk_misuse']
        narrower_threat_not_via_misuse_met = (flop_ >= narrower_threat and not is_deliberate_misuse_ and variables['narrower_threat_is_xrisk_no_misuse'])
        pasta_met = flop_ >= pasta
        if narrower_threat_via_misuse_met or narrower_threat_not_via_misuse_met or pasta_met:
            if verbosity and not state['compute_needs_announced']:
                if pasta_met:
                    threat = 'PASTA'
                    flop_needs = pasta
                elif narrower_threat_via_misuse_met:
                    threat = 'narrower threat via misuse'
                    flop_needs = narrower_threat
                elif narrower_threat_not_via_misuse_met:
                    threat = 'narrower threat not via misuse'
                    flop_needs = narrower_threat
                print('{}: Compute needs for x-risk via {} met ({} vs. {})'.format(y, threat, round(flop_, 1), round(flop_needs, 1)))
                state['compute_needs_announced'] = True

            if not state['initial_delays_calculated']:
                nonscaling_countdown = calculate_nonscaling_delay(y, variables['delay'], variables, verbosity > 0)['nonscaling_countdown']
                state['total_delay'] += nonscaling_countdown
                state['initial_delays_calculated'] = True

            if state['total_delay'] > 0:
                if verbosity > 1:
                    print('-- Compute needs for AI met but {} year nonscaling delay remains'.format(int(round(state['total_delay']))))
                state['total_delay'] -= 1
            else:
                if is_deliberate_misuse_:
                    # TODO: Maybe deliberate misuse by the US would be ok-ish? Could depend on "who gets there first"
                    state = attempt_to_avert_misaligned_tai(state, variables, y, verbosity, intentional_misuse=True)
                else:
                    state = tai_alignment(y, state, variables, verbosity)
                    # TODO: may depend on subtle misalignment
                    if (('fully_aligned' in state['tai_alignment_state'] and p_event(variables, 'p_know_aligned_ai_is_aligned', verbosity)) or
                        ('fully_aligned' not in state['tai_alignment_state'] and not p_event(variables, 'p_know_misaligned_ai_is_misaligned', verbosity))):
                        state = deploy_tai(y, state, variables, verbosity)
                    else:
                        if not coordinate_against_deployment(y, state, variables, verbosity):
                            state = deploy_tai(y, state, variables, verbosity)
                        elif verbosity:
                            print('{}: ...coordinated to not deploy TAI'.format(y))

    return state
