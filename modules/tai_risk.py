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
    solved_alignment = p_event(variables['p_alignment_solved'](y - variables['CURRENT_YEAR'],
                                                               first_attempt=state['tai_type'] is None),
                               'p_alignment_solved',
                               verbosity)
    solved_subtle_alignment = p_event(variables, 'p_subtle_alignment_solved', verbosity) if solved_alignment else False
    fully_aligned_by_default = variables['aligned_by_default'] and solved_subtle_alignment

    if fully_aligned_by_default:
        state['tai_alignment_state'] = 'fully_aligned_by_default'
    elif solved_alignment and solved_subtle_alignment:
        state['tai_alignment_state'] = 'fully_aligned_by_work'
    elif variables['aligned_by_default'] and not fully_aligned_by_default and solved_alignment and solved_subtle_alignment:
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
    if state['tai_alignment_state'] == 'fully_aligned_by_default':
        if verbosity:
            print('{}: ...Achieved aligned TAI (aligned by default)'.format(y))
        state['tai_type'] = 'aligned_agent'
        state['tai'] = True
        state['tai_year'] = y

    elif state['tai_alignment_state'] == 'fully_aligned_by_work':
        if verbosity:
            print('{}: ...Achieved aligned TAI (aligned via work, {} attempt)'.format(y, 'first' if state['tai_type'] is None else '2nd+'))
        state['tai_type'] = 'aligned_agent'
        state['tai'] = True
        state['tai_year'] = y

    elif state['tai_alignment_state'] == 'subtly_misaligned':
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
    if state['tai'] and state['tai_type'] != 'abandoned' and state['tai_type'] != 'aligned_agent':
        if deliberate_misuse(state, variables, verbosity):
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
