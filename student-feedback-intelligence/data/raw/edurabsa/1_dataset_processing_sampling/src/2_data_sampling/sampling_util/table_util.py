import pandas as pd


# ======================================
# Helper functions & illustrations
# ======================================

def resort_by_length_bin(df, ascending=True):
    """ resort dataframe by the 'length_bin' column starting value (lower bound)"""

    df.loc[:, 'temp'] = df.length_bin.astype(str).str.split(',').str[0].str.replace('(','').str.strip().astype(
        'float64')
    df = df.sort_values('temp', ascending=ascending)
    df = df.drop(columns='temp')
    return df


def count_table(df, target_cols, sort_by=None, ascending=True):
    col_list = target_cols if isinstance(target_cols, list) else [target_cols]
    countdf = df[col_list].value_counts().reset_index()

    if sort_by:
        if sort_by == 'length_bin':
            countdf = resort_by_length_bin(countdf, ascending=ascending)
        else:
            countdf = countdf.sort_values(sort_by, ascending=ascending)

    return countdf


def display_count_table(df, target_cols, sort_by=None, ascending=True):
    return display(count_table(df, target_cols, sort_by=sort_by, ascending=ascending))


def get_count_with_ratio(count_df, df_name=None):
    """ Only takes count_table() output type of df """

    if 'length_bin' in count_df.columns:
        count_df.length_bin = count_df.length_bin.astype(str)

    count_df.loc[:, '%'] = count_df['count'].div(count_df['count'].sum())

    if df_name:
        count_df.insert(loc=0, column='df_name', value=df_name)
    # add total row
    count_df.loc[len(count_df)] = count_df.sum(numeric_only=True, axis=0)

    count_df = count_df.fillna('')

    return count_df
# ===================================================
# count_table(source_df,'length_bin',sort_by='length_bin')

def compare_source_and_sample(source_df, sample_df, col_list, sort_by, ascending=True):
    source_rating_count = get_count_with_ratio(count_table(source_df, col_list, sort_by=sort_by, ascending=ascending),
                                               df_name='source_df')
    sample_rating_count = get_count_with_ratio(count_table(sample_df, col_list, sort_by=sort_by, ascending=ascending),
                                               df_name='sample_df')

    df = pd.concat([source_rating_count, sample_rating_count], sort=False, axis=1)

    return df
# ===================================================
# get_count_with_ratio(count_table(source_df, 'length_bin', sort_by='length_bin', ascending=True),df_name='source_df')
# compare_source_and_sample(source_df, sample_df, ['rating_cat', 'rating_bin'], sort_by='count')


def get_sample_distribution(input_df, col_list, index_col, column_col, value_col, aggfunc):

    descriptor = input_df[col_list].fillna('NA').pivot_table(index=index_col, columns=column_col, values=value_col,
                                                             aggfunc=aggfunc).reset_index()
    if index_col=='length_bin':
        descriptor = resort_by_length_bin(descriptor, ascending=True)

    descriptor.loc[:, 'TOTAL'] = descriptor.sum(numeric_only=True, axis=1)

    for each in [i for i in descriptor.columns.values if 'review' in i]:
        descriptor.insert(loc=descriptor.columns.get_loc(each) + 1, column=f'{each} (%)',
                          value=descriptor[each].div(descriptor[each].sum()).round(4) * 100)

    descriptor.loc[:, 'TOTAL (%)'] = descriptor.TOTAL.div(descriptor.TOTAL.sum()).round(4) * 100
    descriptor.loc[len(descriptor)] = ['TOTAL'] + list(descriptor.sum(numeric_only=True, axis=0))

    return descriptor
# =====================================
# get_sample_distribution(input_df, ['type', 'length_bin', 'id'], 'length_bin', 'type', 'id', 'count')
