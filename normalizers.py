def standartize(df, z_standartization_mapping):
    """Provides z-standartization for given dataframe

    :param df - dataframe to standartize
    :param z_standartization_mapping - dictionary with features to standartize, represented as keys.
    Values - helper-dictionary: {'avg': , 'std': } for each feature

    :returns dataframe with standartized features"""
    for feature, mapping in z_standartization_mapping.items():
        feature_column = feature.split('_')
        feature_column.insert(len(feature_column) - 1, 'stand')
        column = ('_').join(feature_column)
        df[column] = df[feature].apply(lambda val: (val - mapping.get('avg')) / mapping.get('std'))

    return df


def normalize_data_wrapper(df, normalize_function, normalize_function_params):
    """wrapper for normalizing dataframe
    :param df - dataframe
    :param normalize_function - normalization function, e.x. z-score standartization
    :param normalize_function_params - parameters for normalization function, e.x. - helper-dict for normalization

    :returns normalized dataframe
    """
    return normalize_function(df, **normalize_function_params)