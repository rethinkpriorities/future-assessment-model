def check_for_time_of_perils(y, state, variables, verbosity):
    if state['category'] == 'aligned_tai' and not state['time_of_perils_end_calculated']:
        if p_event(variables, 'tai_ends_time_of_perils', verbosity):
            if verbosity:
                print('{}: Aligned TAI conclusively ends the time of perils! Happy future! :D'.format(y))
            state['category'] = 'aligned_tai_ends_time_of_perils'
            state['terminate'] = True
            state['final_year'] = y
        else:
            if verbosity:
                print('{}: Aligned TAI does not end time of perils'.format(y))
            state['category'] = 'aligned_tai_does_not_end_time_of_perils'
        state['time_of_perils_end_calculated'] = True

    return state
