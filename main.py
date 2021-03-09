import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    
    TOMATOES_PATH = 'rotten_tomatoes_movies.csv'
    # TMDB_PATH = 'tmdb_5000_movies.csv'
    # IMDB_RATINGS_PATH = 'title.ratings.tsv.gz'
    # IMDB_TITLES_PATH = 'title.akas.tsv.gz'
    tomatoes_df = read_data(TOMATOES_PATH)
    # tmdb_df = read_csv_data(TMDB_PATH)

    critics_vs_audience(tomatoes_df)
    # find_average_runtime_each_genre(tomatoes_df)

    # TMDB_MOVIES_PATH = 'tmdb_5000_movies.csv'
    # tmdb_df = read_csv_data(TMDB_MOVIES_PATH)

    # print(join_tables(tomatoes_df, tmdb_df))  
    
    

def read_data(path):
    if path.split('.')[1] == 'csv':
        return pd.read_csv(path)
    if path.split('.')[1] == 'tsv':
        return pd.read_csv(path, sep='\t')


def critics_vs_audience(raw_data):
    data = raw_data[['genres', 'tomatometer_rating', 'audience_rating']]
    new_data = data.copy()
    new_data['genres'] = new_data['genres'].str.split(',')
    new_data = new_data.explode('genres')
    new_data = new_data.groupby(['genres']).mean()
    new_data = new_data.reset_index()
    sns.set()
    scatter_plot = sns.relplot(data=new_data, x="tomatometer_rating", 
                               y="audience_rating", hue='genres', 
                               kind='scatter', style='genres', 
                               s=50, height=10)
    plt.title('Genres Critics Average Rating vs. Audience Average Rating', weight='bold')
    scatter_plot.set(xlabel='Critics Rating', ylabel='Audience Rating')
    
    plt.plot(range(0, 100), range(0, 100), 'black', alpha=0.2)
    plt.savefig('critics_vs_audience.png', bbox_inches='tight')


def find_average_runtime_each_genre(raw_data):
    """
    Given a data frame find the average movie length from 2000 to present
    """
    data = raw_data[['movie_title', 'streaming_release_date', 'genres', 'runtime']]

    # step 1: drop out rows with no release date
    data = data.dropna(subset=['streaming_release_date'])
    recent_movie = data[data['streaming_release_date'].apply(lambda x :int(x.split('-')[0]) >= 2000)]
    recent_movie = recent_movie.copy()

    # step 2: data processing. Convert genres string attribute to a list
    recent_movie['genres'] = recent_movie['genres'].str.split(',')
    recent_movie = recent_movie.explode('genres')

    # step 3: aggregate the genres average runtime
    recent_movie = recent_movie[['movie_title', 'genres', 'runtime']].groupby(['genres']).mean()

    # step 4: plot the figure
    sns.set()
    plt.figure(figsize=(18,12))
    barplot = sns.barplot(x=recent_movie['runtime'], y=recent_movie.index, palette='magma')
    barplot.set_title('Average Movie Duration from 2000 to Present', fontsize=18, weight='bold')
    barplot.set_xlabel('Movie Duration', fontsize=18, weight='bold')
    barplot.set_ylabel('Genres', fontsize=18, weight='bold')
    plt.savefig('average_runtime_each_genre.png')

def valuable_cast(raw_data):
    pass



  
def join_tables(tomatoes_df, tmdb_df):
    tomatoes_df = tomatoes_df[['movie_title', 'directors']]
    tmdb_df = tmdb_df[['budget', 'revenue', 'title', 'vote_average', 'vote_count']]
    return tomatoes_df.merge(tmdb_df, left_on='movie_title', right_on= 'title', how='left').dropna()



if __name__ == "__main__":
    main()
