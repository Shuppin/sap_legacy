types = [int, float, int, int, float]
values = [1, 1.0, 2.3, 2, 3]

def rearrange_values(spec_list, value_list):

    if not len(spec_list) == len(value_list):
        return None

    rearranged_values = [None]*len(value_list)
    spec_occurance_counter = {}
    value_occurance_counter = {}

    for value in spec_list:
        if spec_occurance_counter.get(value) is None:
            spec_occurance_counter[value] = 1
        else:
            spec_occurance_counter[value] += 1

    for value in value_list:
        if value_occurance_counter.get(type(value)) is None:
            value_occurance_counter[type(value)] = 1
        else:
            value_occurance_counter[type(value)] += 1
        try:
            rearranged_values[[i for i, n in enumerate(spec_list) if n == type(value)][value_occurance_counter[type(value)]-1]] = value
        except IndexError:
            return None

    if not spec_occurance_counter == value_occurance_counter:
        return None

    return tuple(rearranged_values)

print(rearrange_values(types, values))