import numpy as np

def get_percent_steered_to_param_mapping(df):
    """Given a dataframe with columns 'behavior', 'percent_steered', and 'varying_variable_value',
    returns a dictionary mapping each behavior to another dictionary that maps percent_steered
    to varying_variable_value.
    """
    mapping = {
        behavior: dict(zip(df_b["percent_steered"], df_b["varying_variable_value"]))
        for behavior, df_b in df.groupby("behavior")
    }
    return mapping

def filter_mapping(mapping, number_settings, behavior_mapping=None):
    """
    Filters a mapping of behavior -> {percent_steered: value} to a smaller
    set of roughly equally spaced percent_steered values, optionally limited
    by per-behavior maximums.

    Args:
        mapping (dict): {behavior: {percent_steered: value}}
        number_settings (int): total number of percent_steered values to keep
                               (includes min and max)
        behavior_mapping (dict, optional): {behavior: max_percent_steered}.
            If provided, for each behavior, only consider percent_steered values
            <= max_percent_steered.

    Returns:
        dict: filtered mapping in the same format
    """
    filtered = {}

    for behavior, ps_map in mapping.items():
        original_ps = np.array(sorted(ps_map.keys()))

        # If a cap is given for this behavior, apply it
        max_ps_allowed = (
            behavior_mapping[behavior]
            if behavior_mapping and behavior in behavior_mapping
            else original_ps[-1]
        )

        # Keep only points up to that maximum
        eligible_ps = original_ps[original_ps <= max_ps_allowed]

        if len(eligible_ps) <= number_settings:
            filtered[behavior] = {ps: ps_map[ps] for ps in eligible_ps}
            continue

        # Always include min and max within allowed range
        min_ps, max_ps = eligible_ps[0], eligible_ps[-1]
        targets = np.linspace(min_ps, max_ps, number_settings)

        # For each target, pick closest available percent_steered
        selected_ps = []
        for t in targets:
            idx = np.argmin(np.abs(eligible_ps - t))
            selected_ps.append(eligible_ps[idx])

        # Remove duplicates (possible if ps values are sparse)
        selected_ps = sorted(set(selected_ps))

        filtered[behavior] = {ps: ps_map[ps] for ps in selected_ps}

    return filtered
