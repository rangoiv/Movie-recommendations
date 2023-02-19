from numpy import double
import streamlit as st
import pandas as pd
from PIL import Image
import requests
import io

if st.session_state == {}:
    st.session_state = dict()
rating_list = st.session_state

st.set_page_config(
    page_title="Rate movies",
    page_icon="intelligence.jpg",
)


@st.cache_data(persist=True, ttl=86400)
def get_metadata():
    return pd.read_csv("movies_metadata.csv", low_memory=False)


@st.cache_data(persist=True, ttl=86400)
def get_available_movies():
    _ratings = pd.read_csv("../datasets/movies/ratings_small.csv", low_memory=False)
    _avaliable_movies = set()
    for _rating in _ratings["movieId"]:
        _avaliable_movies.add(_rating)
    return _avaliable_movies


st.title("Movies ratings")
number_of_rows = 3

no_image = Image.open(".\\image_not_available.jpg")
available_movies = get_available_movies()
data = get_metadata()
indexes = [str(d).count('-') == 0 and int(d) in available_movies for d in data["id"]]
data = data[indexes]
columns = data.columns

# st.write(columns)

movie = data[['id', 'title', 'poster_path', 'imdb_id', 'popularity']]

# movie[] = movie[movie['popularity'] == "Beware Of Frost Bites"]
# movie['popularity'] = movie['popularity'].replace(['Beware Of Frost Bites'], '0.0')
movie = movie[movie['popularity'] != 'Beware Of Frost Bites']

movie['popularity'] = movie['popularity'].astype(float)

# st.write(movie)

movie = movie.sort_values(by=['popularity'], ascending=False)

movie = movie.head(200)
movie = movie.sample(n=20)

i = 0
j = 0
with st.form("Form"):
    col1, col2, col3 = st.columns(3, gap="large")

    for col_num, col in enumerate([col1, col2, col3]):
        with col:
            while i < number_of_rows * (col_num+1):
                try:
                    title = movie.iloc[j]['title']
                    response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[j]['poster_path'])
                    if response.status_code == 200:
                        st.image(response.content, width=200)
                    else:
                        st.image(no_image, width=200)
                    st.text(title)
                    st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[j]['imdb_id'] + ")")
                    rating = st.slider("Ocijeni film: ", 0.0, 5.0, step=0.5, key=i)

                    rating_list[movie.iloc[j]['id']] = rating
                    # st.write(st.session_state)
                    i = i + 1
                    j = j + 1
                except:
                    print("Loš poster")
                    j = j + 1

    submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state = rating_list
    # st.write(st.session_state)
