import math

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

@st.cache_data(persist=True, ttl=86400)
def get_data():
    data = pd.read_csv("movies_metadata.csv", low_memory=False)
    model_path = "../factorization_movies_model.pkl"
    obj = FactorizationRatingsAprox.from_file(model_path)
    return data, obj

no_image = Image.open(".\\image_not_available.jpg")


st.title("TOP 10 RECOMMENDED MOVIES")

my_ratings = st.session_state


movies_ , obj = get_data()

movies = movies_[['id', 'title', 'poster_path', 'imdb_id', 'popularity']]


ratings = []
for key in my_ratings.keys():
    if my_ratings[key]!=0.0 and int(key)<=163950:
        list =[int(key), my_ratings[key]]
        ratings.append(list)

print("Your ratings: ", ratings)

if ratings != []:
    recommended_ratings = obj.evaluate(ratings)
    recommended = []
    for rating in recommended_ratings:
        movie_id = rating[1]
        score = rating[0]
        people_watched = obj.average_ratings[movie_id][1]
        average_score = obj.average_ratings[movie_id][0] / people_watched
        dif = score-average_score
        flip = -1 if dif < 0 else 1
        new_score = score + flip * abs(dif)**2 * math.log(people_watched**0.5)
        recommended.append([new_score, score, movie_id])
    recommended.sort(reverse=True)
    recommended = [[rating[2], rating[0], rating[1]] for rating in recommended]

    print("Top 10 recomended: ", recommended[:10])
    j=1
    i=0
    while j<11:
        movie = movies[movies['id'] == str(recommended[i][0])]
        try:
            title = movie.iloc[0]['title']
            movie_id = recommended[i][0]
            print(j, movie_id, recommended[i][2], title, obj.average_ratings[movie_id],
                  obj.average_ratings[movie_id][0] / obj.average_ratings[movie_id][1])
            st.header(str(j) + ". " + title)
            st.text("Rating: " + str(round(recommended[i][1],2)))
            response = requests.get("https://image.tmdb.org/t/p/original" + movie.iloc[0]['poster_path'])
            if response.status_code == 200:
                st.image(response.content, width=125)
            else:
                st.image(no_image, width=125)
            st.write(" Watch [trailer](https://www.imdb.com/title/" + movie.iloc[0]['imdb_id'] + ")")
            j=j+1
            i=i+1
        except:
            print(f"Nema filma {recommended[i][0]}")
            i=i+1
    print()
else:
    st.header("Please rate some movies. :back:")