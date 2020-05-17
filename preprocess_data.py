import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
from normalizers import normalize_data_wrapper, standartize


def create_features_columns(df, features_columns=['features']):
    """Parse features from string representation into dataframe's columns

    :param df - dataframe
    :param features_columns - list of columns in df, that are string representation of features,
    each column for each feature code

    :returns dataframe with formed columns (int type), that represents each feature in string
    """
    unique_features_codes = df[features_columns].apply(
        lambda x: x.str.split(',')[0]).apply(lambda x: x[0]).unique()

    features_columns_mapping = {}
    for feature, code in zip(features_columns, unique_features_codes):
        code = df[feature].str.split(',').apply(lambda x: x[0]).unique()[0]
        qty_features = df[feature].str.split(',').apply(lambda x: x[1:]).apply(len).max()

        tmp_list = ['feature_{code}_{index}'.format(code=code, index=index)
                    for index in range(qty_features)]

        features_columns_mapping[feature] = {'code': code, 'features_list': tmp_list}

    for feature_column, mapping in tqdm(features_columns_mapping.items()):
        for index, feature in enumerate(mapping.get('features_list')):
            df[feature] = df[feature_column].str.split(',').apply(lambda x: x[1:][index]).astype(int)

    return df, features_columns_mapping


def create_z_stand_mapping(df, features_mapping):
    """Creates helper-dictionary for z-standartization for each feature and for each feature code

    :param df - dataframe
    :param features_mapping - dictionary, that represents feature codes as keys and list of it's features as values

    :returns helper-dictionary for each feature in form: {feature: {'avg': , 'std': }}
    """
    z_standartization_mapping = defaultdict(dict)

    for features_column, mapping in features_mapping.items():
        for feature in tqdm(chain(*list(mapping.values())[1:])):
            mean_tmp, std_tmp = df[feature].mean(), df[feature].std()
            z_standartization_mapping[feature] = {'avg': mean_tmp,
                                                  'std': std_tmp
                                                  }
    return z_standartization_mapping


def get_max_indexes(df, features_mapping):
    """Calculates index of maximum feature for each id in dataframe and related features

    :param df - dataframe
    :param features_mapping - dictionary, that represents feature codes as keys and list of it's features as values

    :returns dataframe with calculated features"""
    for feature_column, mapping in tqdm(features_mapping.items()):
        code = mapping.get('code')
        features_list = mapping.get('features_list')
        column_title = 'max_feature_{code}_index'.format(code=code)
        df[column_title] = df[features_list].apply(lambda x: x.argmax(), axis=1)

        df['max_feature_{code}_abs_mean_diff'.format(code=code)] = df[[column_title] + features_list].apply(
            lambda x: x[features_list].max() - df[features_list[x[column_title]]].mean(), axis=1)

    return df


def preprocess(df):
    """Wrapper function for data processing

    :param df - dataframe to preprocess

    :returns preprocessed dataframe"""
    data, features_mapping = create_features_columns(df)
    z_standartization_mapping = create_z_stand_mapping(data, features_mapping)

    data = get_max_indexes(data, features_mapping)
    data = normalize_data_wrapper(data, standartize, {'z_standartization_mapping': z_standartization_mapping})

    return data


def main(filename='test.tsv', save=True, path='data/'):
    """main function in script, preprocess data and saves it into file

    :param filename - file with raw data
    :param path - absolute path to file
    :param save - boolean parameter, that determines whether we need to save the results

    :returns preprocessed dataframe"""
    data = pd.read_csv(path+filename, sep='\t')
    preprocessed_data = preprocess(data)

    if save:
        preprocessed_data.to_csv('{path}preprocessed_{filename}'.format(path=path, filename=filename), sep='\t',
                                 index=False)

    return preprocessed_data


main()