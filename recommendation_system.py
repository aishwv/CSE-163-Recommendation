import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
'''
Recommendation system class is build by the given
rotten tomatoes dataset, which recommend movies based on
the similarity of movies and also add some favorite
'''


class Recommendation_System:

    def __init__(self, data, directors):
        '''
        Read movies from dataset and store necessary data for
        future recommendation. Including similarity matrix
        and research result from "movie_analysis" module
        '''
        movies = data[['movie_title', 'genres', 
                       'directors', 'actors', 
                       'production_company']]

        movies = movies.fillna("")
        self._create_tags(movies)
        self._movies = movies
        self._similarity_matrix = self._create_cosine_matrix()
        self._mapper = pd.Series(self._movies.index,
                                 index=self._movies['movie_title'])
        self._directors = directors['directors'].tolist()

    def _create_tags(self, movies):
        '''
        Private helper function
        Create a new column tags
        '''
        movies['tags'] = (movies['genres'] + ", " +
                          movies['directors'] + ", " +
                          movies['actors'] + ", " +
                          movies['production_company'])

    def _create_cosine_matrix(self):
        '''
        Private helper function
        Create a cosine matrix by using
        Tf-idf vectorizer.
        '''
        tf_idf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                 min_df=1, stop_words='english')
        tfidf_matrix = (tf_idf
                        .fit_transform(self._movies['tags'])
                        .astype('float32'))
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    def _matching(self, search_movie, mapper):
        '''
        Private helper function
        Given the search keywords, return the index
        of the most similar movies of the movies database
        '''
        match_tuple = []
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), search_movie.lower())
            if ratio >= 80:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if len(match_tuple) == 0:
            return
        else:
            print(f'Found possible matches in our movies Library:' +
                  f'{[x[0] for x in match_tuple]}\n')
            return (match_tuple[0][0], match_tuple[0][1])

    def recommend(self, search_movie, movies_number=5):
        '''
        Given the search movie,
        Recommend movies based on the similarites of other movies
        '''
        _, idx = self._matching(search_movie, self._mapper)

        if idx is None:
            return
        else:
            sim_scores = list(enumerate(self._similarity_matrix[idx]))
            sim_scores.pop(idx)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[:movies_number]
            movie_indices = [i[0] for i in sim_scores]
            similarity = [i[1] for i in sim_scores]

            return self._calculate_score(movie_indices, similarity)

    def _calculate_score(self, movie_indices, similarity):
        '''
        Private helper function, given movie indices and simiarity value
        calculate the final score and print it out in descending order
        '''
        movies_list = []

        for index, basic_score in zip(movie_indices, similarity):
            score = basic_score
            directors = self._movies['directors'].iloc[index]
            movie_name = self._movies['movie_title'].iloc[index]
            directors = directors.split(',')
            directors = [director.strip() for director in directors]

            for director in directors:
                if director in self._directors:
                    score += 1

            movies_list.append((movie_name, score))
        movies_list.sort(key=lambda x : x[1], reverse=True)

        for movie, score in movies_list:
            print(f'Movie: {movie}, score {score}')

