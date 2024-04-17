# AD Data Recommendation System 

## Introduction

Welcome to the documentation for the AD Data Recommendation System powered by LightFM. This document provides an overview of the system, its components, and how to use it effectively.

## Overview

The AD Data Recommendation System is designed to provide personalized recommendations to users based on their interactions with advertisements. It utilizes the LightFM library, a hybrid recommendation model incorporating collaborative and content-based filtering techniques. The system considers features like visitorEmail, ad_id, and ClickOrNot to make predictions and generate recommendations.

## Components

### 1. Data Preparation

The first step is to prepare the data, involving collecting user interactions with advertisements (clicks or views) and structuring it for model training. Data includes mulitple features but used only three features for now. 

- `visitorEmail`: The user's email address.
- `ad_id`: The unique advertisement identifier.
- `ClickOrNot`: Binary indicator (0 or 1) representing ad interaction.

### 2. Model Training

Next, train the LightFM model. Fit it to the training data, optimizing parameters to minimize prediction error. The model learns patterns to predict user-advertisement interactions accurately.


### Model Parameters

- `num_factors`: 3
- `num_epochs`: 150
- `learning_rate`: 0.05
- `loss`: 'warp'
- `item_alpha`: 0.0001
- `user_alpha`: 0.0001


### 3. Prediction

After training, use the model for predictions. Given a user and ad, the model computes a predicted score indicating interaction likelihood. It considers user's past interactions and ad characteristics.

## Evaluation Metrics

![alt text](<Screenshot from 2024-04-02 15-21-29.png>)

### 4. Recommendation Logic

The model can generate top recommendations for each user based on predicted scores. Recommendations are ranked, with highest-scoring ads at the top. Typically, N top recommendations are generated for each user.

#### Existing Users: 
For existing users (e.g., pramodyop@mail.com, kalipopup@mail.com), the system recommends the top 10 ads based on their past interactions.



![alt text](<Screenshot from 2024-04-09 14-47-12.png>)



![alt text](<Screenshot from 2024-04-09 14-45-28.png>)


#### New Users: 
For new users, the system recommends a random ad.
![alt text](<Screenshot from 2024-04-09 14-48-41.png>)