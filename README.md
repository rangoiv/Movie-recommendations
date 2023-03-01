# Tensor factorization and movie reccomendation

This project is about exploring movie reccomendation algorithms based
on collaborative filtering.
First part is about implementing tensor factorization algorithm based 
on work [1].
Secondly we compared it to other available algorithms suchs as matrix
factorization from
[here](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b#:~:text=Matrix%20factorization%20is%20a%20collaborative,both%20item%20and%20user%20entities).
We also compared it to k-nearest neighbours method and deep learnng
models from [fastai](https://www.fast.ai/) library.
The comparisons were made on
[kaggle movie dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv).

Thirly, we created a simple application for recommending movies. It also
gets data from the same dataset, but with tensor factorisation algorithm
running to get better predictions.

It by no means works as well as Netflix :), but it is a simple example on
how to use these methods for your applications. Of course, it only shows how
complex these recommendations algorithms must be to get perfect and why
is Netflix making billions :O

## Screenshots

Here are screenshots of movie rating screen and movie recommendation screen.

![screen-1.jpeg](images%2Fscreen-1.jpeg)

![screen-2.jpeg](images%2Fscreen-2.jpeg)

## Code structure

All the code for algorithms are in this main folder: some files for
tensor factorisation and K-nearest neighbours.

There is also a simple implementation for Sparse matrix needed to store
big chunks of tabular data where most values are zeroes.

## Requirements

To run application you need:

1. Python installed 
2. Install [streamlit](https://docs.streamlit.io/library/get-started/installation)

## How to run

1. Open terminal in `streamlit` folder
2. Run `streamlit run 📈_Rate_movies`

## Jupyter notebooks

To test how algorithms work, open notebooks in `notebook` folder.
There are two files, one is for fastai and the other is for tensor
factorization algorithm.

You need Jupyter notebook installed.


## Citations 
1. Alexandros Karatzoglou, Xavier Amatriain, Linas Baltrunas, and Nuria
Oliver. “Multiverse recommendation: n-dimensional tensor factorization for
context-aware collaborative filtering