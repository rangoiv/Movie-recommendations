from numpy import double
import streamlit as st
import pandas as pd
from PIL import Image
import requests
import io

if st.session_state=={}:
    st.session_state = dict()
    rating_list=dict()
rating_list = st.session_state
st.set_page_config(
    page_title="Rate movies",
    page_icon="intelligence.jpg",
)


st.title("Movies ratings")

image = Image.open(".\image_not_available.jpg")
data = pd.read_csv("movies_metadata.csv")
columns = data.columns

# st.write(columns)


movie = data[['id', 'title', 'poster_path', 'imdb_id', 'popularity']]

movie.drop(movie[movie['popularity'] == "Beware Of Frost Bites"].index, inplace = True)

movie['popularity'] = movie['popularity'].astype(float)



# st.write(movie)

movie = movie.sort_values(by=['popularity'], ascending=False)

movie = movie.head(100)
movie = movie.sample(n=12)



with st.form("Form"):
    
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        for i in range (4):
            st.header(movie.iloc[i]['title'])
            response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[i]['poster_path'])
            if response.status_code == 200:
                st.image(response.content)
            else:
                st.image(image)
            st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[i]['imdb_id'] + ")")
            rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)

            rating_list[movie.iloc[i]['id']] = rating
            # st.write(st.session_state)


    with col2:
        for i in range (4,8):
            st.header(movie.iloc[i]['title'])
            response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[i]['poster_path'])
            if response.status_code == 200:
                st.image(response.content)
            else:
                st.image(image)
            st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[i]['imdb_id'] + ")")
            rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)
            rating_list[movie.iloc[i]['id']]= rating


    

    with col3:
        for i in range (8, 12):
            st.header(movie.iloc[i]['title'])
            response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[i]['poster_path'])
            if response.status_code == 200:
                st.image(response.content)
            else:
                st.image(image)
            st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[i]['imdb_id'] + ")")
            rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)
            rating_list[movie.iloc[i]['id']]= rating


    submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state = rating_list
    st.write(st.session_state)

