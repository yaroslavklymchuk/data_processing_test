import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
from normalizers import normalize_data_wrapper, standartize


def create_features_columns(df, features_columns=['features']):
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
    z_standartization_mapping = defaultdict(dict)

    for features_column, mapping in features_mapping.items():
        for feature in tqdm(chain(*list(mapping.values())[1:])):
            mean_tmp, std_tmp = df[feature].mean(), df[feature].std()
            z_standartization_mapping[feature] = {'avg': mean_tmp,
                                                  'std': std_tmp
                                                  }
    return z_standartization_mapping


def get_max_indexes(df, features_mapping):
    for feature_column, mapping in tqdm(features_mapping.items()):
        code = mapping.get('code')
        features_list = mapping.get('features_list')
        column_title = 'max_feature_{code}_index'.format(code=code)
        df[column_title] = df[features_list].apply(lambda x: x.argmax(), axis=1)

        df['max_feature_{code}_abs_mean_diff'.format(code=code)] = df[[column_title] + features_list].apply(
            lambda x: x[features_list].max() - df[features_list[x[column_title]]].mean(), axis=1)

    return df


def preprocess(df):
    data, features_mapping = create_features_columns(df)
    z_standartization_mapping = create_z_stand_mapping(data, features_mapping)

    data = get_max_indexes(data, features_mapping)
    data = normalize_data_wrapper(data, standartize, {'z_standartization_mapping': z_standartization_mapping})

    return data


def main(filename='data/test.tsv', save=True):
    data = pd.read_csv(filename, sep='\t')
    preprocessed_data = preprocess(data)

    if save:
        preprocessed_data.to_csv('data/preprocessed_{}'.format(filename), sep='\t')

    return preprocessed_data


