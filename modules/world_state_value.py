def value_of_world_state(world_state, variables):
    category = world_state['category']
    terminate = world_state['terminate']
    if terminate:
        final_year = world_state['final_year']
        years_lived = final_year - variables['CURRENT_YEAR']
        total_value = variables['total_present_value'] + variables['total_additional_value_per_year'] * years_lived
    else:
        total_value = variables['total_present_value'] + variables['total_additional_value_per_year'] * variables['years_to_consider']

    # if category == 'aligned_tai':
        # TODO: Additional QALY benefit from TAI?
    # elif category in ['xrisk_subtly_unaligned_tai', 'xrisk_tai_misuse']:
        # TODO: value of subtle misalignment should be >0 ...though also possibly <0
        # TODO: States worse than 0?
    # TODO: Loss of value from catastrophes and wars?
    return total_value
