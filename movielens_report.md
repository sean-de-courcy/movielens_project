Movielens Final Project
================
Sean Coursey
2/6/2022

## Introduction

This project was completed for the *Data Science: Capstone* course in
HarvardX’s *Data Science Professional Certificate* program on edX. The
general goal of this project was to use a dataset of ten million movie
ratings to create a model which could predict significantly better than
chance what rating any particular user would give any particular movie.
The publicly available dataset was curated by GroupLens, which is a
computer science research lab at the University of Minnesota, Twin
Cities. The data comes from their movielens project–which is an online,
non-commercial movie recommendation system–and includes ratings from
users who have rated at least 20 movies. For this project, HarvardX
provided code for producing a training set of approximately nine million
observations and a validation set of approximately one million, and the
specific goal of the project was to create a model using the training
set which could predict the ratings in the validation set with a root
mean squared error (or rmse) of less than 0.8649. The model described in
this report achieved an rmse of 0.797 by centering the data about the
average ratings of each user and then accounting for the difference from
each user’s average rating using matrix factorization (enabled by the
library *recosystem*).

## Methods

The following libraries were used in this analysis:

``` r
library(tidyverse)
library(caret)
library(data.table)
library(recosystem)
library(Metrics)
```

Firstly, the code provided by HarvardX was used to download the data
from GroupLens and partition it into training and validation data
tables–named `edx` and `validation`, respectively. Each observation in
these data sets includes six variables:

``` r
validation[42]
```

    ##    userId movieId rating  timestamp                  title genres
    ## 1:      8     489    3.5 1115860202 Made in America (1993) Comedy

Matrix factorization is an unsupervised numerical method, which means
the character variables `title` and `genres` are difficult to include
usefully. While key-word analysis of `title` and `genres` could produce
meaningful results which could theoretically be numericized, this
process was deemed far too difficult and memory-intensive to justify for
the marginal benefits it could produce. Similarly, `timestamp`–while
numerical–was reasoned to not include enough useful information to be
worth including in the matrix factorization. Following this reasoning,
the `edx` and `validation` data tables were modified to only include
`userId`, `movieId`, and `rating`. Then, the edx table was partitioned
into a training set (`train`) and a testing set (`test`) so a model
could be created and tested without using the validation set.
