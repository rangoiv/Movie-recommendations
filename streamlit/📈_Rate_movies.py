import streamlit as st
import pandas as pd
from PIL import Image


st.set_page_config(
    page_title="Rate movies",
    page_icon="intelligence.jpg",
)

st.title("Movies ratings")

data = pd.read_csv("movies_metadata.csv")
columns = data.columns

st.write(columns)

movie = data[['id', 'title', 'poster_path']]


movie = movie.sample(n=3)
# st.dataframe(movie)



with st.form("Form"):
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header(movie.iloc[0]['title'])
        st.image("https://image.tmdb.org/t/p/original" + movie.iloc[0]['poster_path'])
        a=st.slider("Ocijeni film: ", 1, 5, key = "1")


    with col2:
        st.header(movie.iloc[1]['title'])
        st.image("https://image.tmdb.org/t/p/original" + movie.iloc[1]['poster_path'])
        a=st.slider("Ocijeni film: ", 1, 5, key = "2")


    

    with col3:
        st.header(movie.iloc[2]['title'])
        st.image("https://image.tmdb.org/t/p/original" + movie.iloc[2]['poster_path'])
        c=st.slider("Ocijeni film: ", 1, 5, key = "3")

    submitted = st.form_submit_button("Submit")

