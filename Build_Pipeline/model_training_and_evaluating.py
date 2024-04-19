import mlflow
import mlflow.pyfunc
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from Build_Pipeline.data_splitting import train_set, test_set

mlflow.set_tracking_uri("http://3.249.112.184:5000/")
mlflow.set_experiment(experiment_id="1")

class LightFMPythonModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict(self, context, model_input):
        user_ids = [row['visitorEmail'] for _, row in model_input.iterrows()]
        item_ids = [row['ad_id'] for _, row in model_input.iterrows()]
        interactions, _ = self.dataset.build_interactions(zip(user_ids, item_ids))
        scores = self.model.predict_rank(interactions, user_ids)
        return scores

class LightFMRecommender:
    def __init__(self, train_set, test_set, num_factors=3, num_epochs=150, learning_rate=0.05,
                 loss='warp', item_alpha=0.0001, user_alpha=0.0001):
        self.train_set = train_set
        self.test_set = test_set
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.model = None
        self.dataset = None

    def fit_model(self):
        # Initialize the LightFM model
        self.model = LightFM(no_components=self.num_factors, loss=self.loss,
                             learning_rate=self.learning_rate, item_alpha=self.item_alpha,
                             user_alpha=self.user_alpha)

        # Create a Dataset object
        self.dataset = Dataset()

        # Fit the dataset on the train_set DataFrame to create the user and item indices
        self.dataset.fit((user for user in self.train_set['visitorEmail']),
                         (item for item in self.train_set['ad_id']))

        # Build the interaction matrix
        interactions, _ = self.dataset.build_interactions(((row['visitorEmail'], row['ad_id']) for index, row in self.train_set.iterrows()))

        # Train the model
        self.model.fit(interactions, epochs=self.num_epochs)
        print("Model trained successfully.")

    def evaluate_model(self, k=10):
        # Filter test set to include only user IDs and item IDs present in the training set
        filtered_test_set = self.test_set[(self.test_set['visitorEmail'].isin(self.train_set['visitorEmail'])) & 
                                           (self.test_set['ad_id'].isin(self.train_set['ad_id']))]

        # Convert the filtered test set to interactions
        test_interactions, _ = self.dataset.build_interactions(((row['visitorEmail'], row['ad_id']) for index, row in filtered_test_set.iterrows()))

        # Calculate precision at k
        precision = precision_at_k(self.model, test_interactions, k=k).mean()

        # Calculate recall at k
        recall = recall_at_k(self.model, test_interactions, k=k).mean()

        print(f"Precision at {k}: {precision}")
        print(f"Recall at {k}: {recall}")

        # Log evaluation metrics
        mlflow.log_metrics({
            f"PrecisionatK": precision,
            f"RecallatK": recall
        })

    # def save_model(self, path):
    #     mlflow.pyfunc.save_model(path, python_model=LightFMPythonModel(self.model, self.dataset))

# Example usage:
if __name__ == "__main__":
    # Assuming train_set and test_set are already defined
    with mlflow.start_run(run_name="Lightfm", nested=True) as run:
        recommender = LightFMRecommender(train_set, test_set)
        recommender.fit_model()
        recommender.evaluate_model()
        # Save the model
        # recommender.save_model("lightfm_model")
