def value_of_world_state(world_state, total_present_value, total_additional_value_per_year, years_to_consider):
	category = world_state['category']
	final_year = world_state['final_year']
	if final_year is None:
		total_value = total_present_value + total_additional_value_per_year * years_to_consider
	else:
		years_lived = final_year - CURRENT_YEAR
		total_value = total_present_value + total_additional_value_per_year * final_year

    # if category == 'aligned_tai':
        # TODO: Additional QALY benefit from TAI?
	# elif category in ['xrisk_subtly_unaligned_tai', 'xrisk_tai_misuse']:
		# TODO: States worse than 0?
	# TODO: Loss of value from catastrophes and wars?
	return total_value
