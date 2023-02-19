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

@st.cache_data
def get_data1():
    data = pd.read_csv("movies_metadata.csv", low_memory=False)
    return data

st.title("Movies ratings")

no_image = Image.open(".\\image_not_available.jpg")
data = get_data1()
columns = data.columns

# st.write(columns)


movie = data[['id', 'title', 'poster_path', 'imdb_id', 'popularity']]

# movie[] = movie[movie['popularity'] == "Beware Of Frost Bites"]
# movie['popularity'] = movie['popularity'].replace(['Beware Of Frost Bites'], '0.0')
movie = movie[movie['popularity'] != 'Beware Of Frost Bites']

movie['popularity'] = movie['popularity'].astype(float)



# st.write(movie)

movie = movie.sort_values(by=['popularity'], ascending=False)

movie = movie.head(100)
movie = movie.sample(n=20)


i=0
j=0
with st.form("Form"):
    
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        while i<4:
            try:
                st.text(movie.iloc[j]['title'])
                response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[j]['poster_path'])
                if response.status_code == 200:
                    st.image(response.content, width=200)
                else:
                    st.image(no_image, width=200)
                st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[j]['imdb_id'] + ")")
                rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)

                rating_list[movie.iloc[j]['id']] = rating
                # st.write(st.session_state)
                i=i+1
                j=j+1
            except:
                print("Loš poster")
                j=j+1


    with col2:
        while i<8:
            try:
                st.text(movie.iloc[j]['title'])
                response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[j]['poster_path'])
                if response.status_code == 200:
                    st.image(response.content, width=200)
                else:
                    st.image(no_image,width=200)
                st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[j]['imdb_id'] + ")")
                rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)
                rating_list[movie.iloc[j]['id']]= rating
                i=i+1
                j=j+1
            except:
                print("Loš poster")
                j=j+1


    

    with col3:
        while i<12:
            try:
                st.text(movie.iloc[j]['title'])
                response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[j]['poster_path'])
                if response.status_code == 200:
                    st.image(response.content, width=200)
                else:
                    st.image(no_image, width=200)
                st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[j]['imdb_id'] + ")")
                rating=st.slider("Ocijeni film: ", 0.0, 5.0, step = 0.5, key = i)
                rating_list[movie.iloc[j]['id']]= rating
                i=i+1
                j=j+1
            except:
                print("Loš poster")
                j=j+1


    submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state = rating_list
    st.write(st.session_state)

