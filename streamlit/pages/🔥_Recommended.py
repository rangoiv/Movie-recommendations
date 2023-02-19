import streamlit as st
import pandas as pd
import sys
import numpy as np
sys.path.append('../')
from tensor_factorization import initialize_factorization, D, fine_tune, Lambda
from tensor_factorization import evaluate
from FactorizationRatingsAprox import FactorizationRatingsAprox
from split_data import split_data
from sparse_array import NDSparseArray
from datetime import datetime
from PIL import Image
import requests


st.set_page_config(
    page_title="Recommended movies",
    page_icon="intelligence.jpg",
)
no_image = Image.open(".\\image_not_available.jpg")


st.title("TOP 10 RECOMMENDED MOVIES")

my_ratings = st.session_state

movies_ = pd.read_csv("movies_metadata.csv", low_memory=False)

movies = movies_[['id', 'title', 'poster_path', 'imdb_id', 'popularity']]


model_path = "../factorization_movies_model.pkl"

obj = FactorizationRatingsAprox.from_file(model_path)

ratings = []
for key in my_ratings.keys():
    if my_ratings[key]!=0.0 and int(key)<=163950:
        list =[int(key), my_ratings[key]]
        ratings.append(list)

print(ratings)

recommended = obj.evaluate(ratings)


j=1
for i in range (10):
    movie = movies[movies['id'] == str(recommended[i][0])]
    
    try: 
        st.header(str(j) + ". " + movie.iloc[0]['title'])
        response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[0]['poster_path'])
        if response.status_code == 200:
            st.image(response.content)
        else:
            st.image(no_image)
        st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[0]['imdb_id'] + ")")
        j=j+1
    except:
        print("Nema filma")
    