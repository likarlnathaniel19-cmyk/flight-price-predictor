import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 1. Load the data you provided
data = {
    'airline': ['AirAsia', 'Cebu Pacific', 'Philippine Airlines', 'AirAsia', 'Cebu Pacific', 'Philippine Airlines', 'AirAsia', 'Cebu Pacific', 'Philippine Airlines', 'AirAsia', 'Cebu Pacific', 'Philippine Airlines'],
    'source': ['Manila', 'Manila', 'Manila', 'Cebu', 'Davao', 'Davao', 'Manila', 'Iloilo', 'Manila', 'Bacolod', 'Manila', 'General Santos'],
    'destination': ['Cebu', 'Davao', 'Cebu', 'Manila', 'Manila', 'Cebu', 'Iloilo', 'Manila', 'Bacolod', 'Manila', 'General Santos', 'Manila'],
    'price': [2500, 3200, 4500, 2600, 3500, 4800, 2800, 3000, 4000, 2700, 3600, 500]
}
df = pd.DataFrame(data)

X = df[['airline', 'source', 'destination']]
y = df['price']

# 2. Create a preprocessor to convert text columns to numbers automatically
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['airline', 'source', 'destination'])
    ]
)

# 3. Combine the preprocessor and the Model into a single Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 4. Train the integrated model
pipeline.fit(X, y)

# 5. Save the entire pipeline as your model file
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("SUCCESS: New model.pkl pipeline created successfully!")