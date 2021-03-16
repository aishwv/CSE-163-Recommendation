import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cse163_utils
from recommendation_system import Recommendation_System


def main():
    TOMATOES_TEST_PATH = 'tomatoes_test.csv'
    TMDB_TEST_PATH = 'tmdb_test.csv'
    JOINT_TABLE_TEST_PATH = 'joint_table_test.csv'
    # load data
    test_tomatoes_df = read_data(TOMATOES_TEST_PATH)
    test_tmdb_df = read_data(TMDB_TEST_PATH)
    test_joint_table_df = read_data(JOINT_TABLE_TEST_PATH)

    # test all function
    test_critcs_vs_audience(test_tomatoes_df)
    test_find_average_runtime_each_genre(test_tomatoes_df)
    test_popular_genre_by_country(test_tmdb_df)
    test_find_valuable_cast(test_joint_table_df)

    TOMATOES_PATH = 'rotten_tomatoes_movies.csv'
    TMDB_PATH = 'tmdb_5000_movies.csv'
    tomatoes_df = read_data(TOMATOES_PATH)
    tmdb_df = read_data(TMDB_PATH)
    joint_table_df = join_tables(tomatoes_df, tmdb_df)

    # plot all research figure and get result
    print("start to plot.................")
    critics_vs_audience(tomatoes_df)
    find_average_runtime_each_genre(tomatoes_df)
    popular_genre_by_country(tmdb_df)
    top_10_directors = find_valuable_cast(joint_table_df)

    print('start to build the recommendation system.......')
    rc = Recommendation_System(tomatoes_df, top_10_directors)
    rc.recommend('Avatar', 20)


def test_critcs_vs_audience(data):
    """
    test the function "critics_vs_audience"
    """
    result = critics_vs_audience(data)

    # test classics movie
    row_data = result.query('genres == "Classics"')
    cse163_utils.assert_equals(row_data.values[0][1], 50)
    cse163_utils.assert_equals(row_data.values[0][2], 65)

    # test action movie
    row_data = result.query('genres == "Action"')
    cse163_utils.assert_equals(row_data.values[0][1], 50)
    cse163_utils.assert_equals(row_data.values[0][2], 100)


def test_find_average_runtime_each_genre(data):
    '''
    test the function "find_average_runtime_each_genre"
    '''
    result = find_average_runtime_each_genre(data)

    # # test classics movie
    row_data = result.query('genres == "Classics"')
    cse163_utils.assert_equals(row_data.values[0][1], 50)

    # test action movie
    row_data = result.query('genres == "Action"')
    cse163_utils.assert_equals(row_data.values[0][1], 100)


def test_popular_genre_by_country(data):
    '''
    test the function "popular_genre_by_country" 
    '''

    result = popular_genre_by_country(data)

    # test country United States
    row_data = result.query('main_country == "United States of America"')
    cse163_utils.assert_equals(row_data.values[0][1], 'Action')
    cse163_utils.assert_equals(row_data.values[0][2], 200)

    # test country Japan
    row_data = result.query('main_country == "Japan"')
    cse163_utils.assert_equals(row_data.values[0][1], 'Drama')
    cse163_utils.assert_equals(row_data.values[0][2], 100)


def test_find_valuable_cast(data):
    '''
    test the function "find_valuable_cast" 
    '''
    result = find_valuable_cast(data)

    # test one director (made up)
    row_data = result.query('directors == "Aishwarya Venkatesh"')
    cse163_utils.assert_equals(row_data.values[0][1], 12345)

    # test another director (made up)
    row_data = result.query('directors == "Dominic Minichillo"')
    cse163_utils.assert_equals(row_data.values[0][1], 100000)


def read_data(path):
    if path.split('.')[1] == 'csv':
        return pd.read_csv(path)


def critics_vs_audience(raw_data):
    '''
    Given data, compare the genre ratings of audience and critics.
    Plot the scatter figure of the ciritcs and audience and returns
    a dataframe with genre, critic ratings and audience rating.
    '''
    data = raw_data[['genres', 'tomatometer_rating', 'audience_rating']]
    new_data = data.copy()
    new_data['genres'] = new_data['genres'].str.split(',')
    new_data = new_data.explode('genres')
    new_data['genres'] = new_data['genres'].str.strip()
    new_data = new_data.groupby(['genres']).mean()
    new_data = new_data.reset_index()
    sns.set()
    scatter_plot = sns.relplot(data=new_data, x="tomatometer_rating",
                               y="audience_rating",
                               kind='scatter', hue='genres',
                               s=100, style='genres',
                               height=10)

    plt.title('Genres Critics Average Rating' +
              'vs. Audience Average Rating',
              weight='bold')
    scatter_plot.set(xlabel='Critics Rating', ylabel='Audience Rating')

    plt.plot(range(0, 100), range(0, 100), 'black', alpha=0.2)
    plt.savefig('critics_vs_audience.png', bbox_inches='tight')
    return new_data


def find_average_runtime_each_genre(raw_data):
    """
    Given a data frame find the average movie length from
    2000 to present. Plot the figure and returns the
    statistical result of average duration of each genre
    in form of DataFrame
    """
    data = raw_data[['streaming_release_date',
                    'genres', 'runtime']]

    # step 1: drop out rows with no release date
    data = data.dropna(subset=['streaming_release_date'])
    recent_movie = data[data['streaming_release_date'].
                        apply(lambda x:int(x.split('-')[0]) >= 2000)]
    recent_movie = recent_movie.copy()

    # step 2: data processing. Convert genres string attribute to a list
    recent_movie['genres'] = recent_movie['genres'].str.split(',')
    recent_movie = recent_movie.explode('genres')
    recent_movie['genres'] = recent_movie['genres'].str.strip()

    # step 3: plot the figure
    sns.set()
    plt.figure(figsize=(18, 12))
    barplot = sns.barplot(x='runtime', y='genres',
                          palette='magma', ci=50,
                          data=recent_movie)
    barplot.set_title('Average Movie Duration from 2000 to Present',
                      fontsize=18,
                      weight='bold')
    barplot.set_xlabel('Movie Duration', fontsize=18, weight='bold')
    barplot.set_ylabel('Genres', fontsize=18, weight='bold')
    plt.savefig('average_runtime_each_genre.png')

    return recent_movie.groupby('genres').describe()


def popular_genre_by_country(raw_data):
    """
    This takes the tmdb 5000 movies dataset
    and finds the most popular movie genre in
    each country. Plot the figures of each country with
    their most popular genre. Return a dataFrame with
    each country, with its main genre and its popularity value.
    """
    popular_genre = raw_data[["popularity", 'genres',
                              'production_countries']]
    # The three relevant coulmns. Popularity is a metric calculated by tmdb
    genre_list = []
    country_list = []
    for i in range(len(popular_genre)):
        genre_list.append(popular_genre.loc[i, "genres"])
        country_list.append(popular_genre.loc[i, "production_countries"])
    # revelant info is in a list of dictionaries
    main_genre = []
    for genres in genre_list:
        genres = eval(genres)
        count = 0
        if len(genres) == 0:
            main_genre.append('N/A')
        for j in genres:
            count += 1
            if count == 1:
                values = j.values()
                values_list = list(values)
                main_genre.append(values_list[1])
    # getting main genre for each movie
    main_country = []
    for countries in country_list:
        countries = eval(countries)
        count = 0
        if len(countries) == 0:
            main_country.append('N/A')
        for k in countries:
            count += 1
            if count == 1:
                values = k.values()
                values_list = list(values)
                main_country.append(values_list[1])
    # getting main production country for each movie
    popular_genre = popular_genre.assign(
        main_genre=pd.Series(main_genre).values)
    popular_genre = popular_genre.assign(
        main_country=pd.Series(main_country).values)
    # adding genre and countries to df as a series
    groupby = popular_genre.groupby(['main_country', 'main_genre'],
                                    as_index=False).popularity.mean().\
        sort_values(['main_country', 'popularity'],
                    ascending=False)
    # avg popularity for country genre combo in order
    final_groupby = groupby.groupby('main_country').head(1)
    # gets most popular genre for each country
    final_groupby = final_groupby.nlargest(20, ['popularity'])
    plot = sns.barplot(x="main_country", y='popularity', hue='main_genre',
                       data=final_groupby, dodge=False)
    # barplot if the results with genre as part of the legend
    plt.xlabel('Country')
    plt.ylabel('Popularity Metric')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.title("Most Popular Genre in Each Country")
    plt.legend(bbox_to_anchor=(.75, 1), loc=2, borderaxespad=0.)
    plt.gcf().subplots_adjust(bottom=0.15)
    fig = plot.get_figure()
    fig.savefig("most_popular_genre_pop.png")
    return final_groupby


def find_valuable_cast(joint_df):
    """

    """
    data_directors = joint_df[['directors', 'revenue']]
    data_directors = data_directors.copy()
    data_directors['directors'] = data_directors['directors'].str.split(',')
    data_directors = data_directors.explode('directors')
    data_directors['directors'] = data_directors['directors'].str.strip()
    data_directors_revenue = (data_directors
                              .sort_values('revenue', ascending=False)
                              .drop_duplicates('directors', keep='first')
                              .head(10))
    sns.set()
    plt.figure(figsize=(18, 12))
    barplot_directors = sns.barplot(x=data_directors_revenue['revenue'],
                                    y=data_directors_revenue['directors'],
                                    palette='viridis')
    barplot_directors.set_title(
        'Top 10 Most Profitable Directors Of All Time', fontsize=18, weight='bold')
    barplot_directors.set_xlabel('Revenue in USD', fontsize=18, weight='bold')
    barplot_directors.set_ylabel('Directors', fontsize=18, weight='bold')
    plt.savefig('valuable_directors.png')
    return data_directors_revenue


def join_tables(tomatoes_df, tmdb_df):
    """

    """
    tomatoes_df = tomatoes_df[['movie_title', 'directors']]
    tmdb_df = tmdb_df[['budget', 'revenue',
                       'title', 'vote_average', 'vote_count']]
    return tomatoes_df.merge(tmdb_df, left_on='movie_title',
                             right_on='title', how='left').dropna()


if __name__ == "__main__":
    main()
