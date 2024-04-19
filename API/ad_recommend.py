import random
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from helpers import filter_data, preprocess_data

# Create an instance of FastAPI
app = FastAPI()

# Number of top items to recommend
top_n = 10
filtered_data = pd.read_csv("filtered_data.csv")

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the recommendation endpoint
@app.get("/recommend/{visitor_email}")
async def recommend_items(visitor_email: str):

    existing_users = filtered_data['visitorEmail'].unique()
    user_id_map = {email: i for i, email in enumerate(existing_users)}
    item_id_map = {item: i for i, item in enumerate(filtered_data['ad_id'].unique())}

    if visitor_email in existing_users:
        
        # Get the items for which the user has not interacted
        user_interacted_items = filtered_data[filtered_data['visitorEmail'] == visitor_email]['ad_id'].unique()
        all_items = filtered_data['ad_id'].unique()
        items_to_recommend = np.setdiff1d(all_items, user_interacted_items)
        # Map user email to integer user ID
        user_id = user_id_map[visitor_email]

        # Map item IDs to integers
        items_to_recommend_ids = [item_id_map[item] for item in items_to_recommend]
        
        

        print(np.array([user_id]*len(items_to_recommend[:449])).shape, np.array(items_to_recommend_ids[:449]).shape)
        recommendations = model.predict(
        user_ids=np.array([user_id]*len(items_to_recommend)),
        item_ids=np.array(items_to_recommend_ids)
    )

        # # Get top-N recommended items
        top_indices = np.argsort(recommendations)[::-1][:top_n]
        top_items = [items_to_recommend[index] for index in top_indices]

        # Print top-N recommended items for the user
        # print(f"Top {top_n} recommended items for user {visitor_email}:")
        for index, item in enumerate(top_items):
            print(f"Item ID: {item}, Predicted Score: {recommendations[index]}")
        return {
            "message":"success",

            "result":[{"ad_id":item,
                       "score":float(recommendations[index])} for index,item in enumerate(top_items)]
        }
    else:
        # If the user is unknown, recommend a random ad
        all_items = filtered_data['ad_id'].unique()
        random_ad = random.choice(all_items)

        return {
            "message": "success",

            "result": [{"ad_id": random_ad}]  # You can set score to None for random recommendation
        }


@app.get("/")
def home():
    return "Hello, Greetings !"

 
# Run the FastAPI application using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5055)

