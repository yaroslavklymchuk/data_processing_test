def standartize(df, z_standartization_mapping):
    for feature, mapping in z_standartization_mapping.items():
        feature_column = feature.split('_')
        feature_column.insert(len(feature_column) - 1, 'stand')
        column = ('_').join(feature_column)
        df[column] = df[feature].apply(lambda val: (val - mapping.get('avg')) / mapping.get('std'))

    return df


def normalize_data_wrapper(df, normalize_function, normalize_function_params):
    return normalize_function(df, **normalize_function_params)