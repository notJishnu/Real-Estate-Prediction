import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix

# --- App Configuration ---
st.set_page_config(page_title="Boston Housing EDA & Prediction", layout="wide")
st.title("Boston Housing Dataset - EDA & Prediction")
st.markdown("This app performs exploratory data analysis and predicts median house values using a Random Forest model.")

# --- Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("BostonHousing.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'BostonHousing.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

# --- 1. Data Inspection ---
st.header("1. Data Loading & Inspection")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Description")
    st.dataframe(df.describe())
with col2:
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

# --- 2. Feature Distribution ---
st.header("2. Feature Distributions")
if st.checkbox("Show Feature Distributions"):
    st.subheader("Charles River Dummy Variable (chas)")
    st.bar_chart(df['chas'].value_counts())

    st.subheader("Histograms of All Features")
    fig, ax = plt.subplots(figsize=(15, 10))
    df.hist(bins=50, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# --- 3. Train-Test Splitting ---
st.header("3. Train-Test Splitting")
split_type = st.radio("Select Splitting Method:", ["Random Split (80/20)", "Stratified Split (by 'chas')"])

if split_type == "Random Split (80/20)":
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    st.write(f"**Training set shape:** {train_set.shape} | **Test set shape:** {test_set.shape}")
else:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['chas']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    st.write(f"**Stratified Training set shape:** {train_set.shape} | **Stratified Test set shape:** {test_set.shape}")

# --- 4. Correlation Analysis ---
st.header("4. Correlation Analysis")
corr_matrix = train_set.corr()
st.write("Correlation with Median House Value (`medv`):")
corr_medv = corr_matrix['medv'].sort_values(ascending=False)
st.dataframe(corr_medv)

# --- 5. Scatter Matrix Visualization ---
st.header("5. Scatter Matrix")
if st.checkbox("Show Scatter Matrix for Top Features"):
    attributes = ["medv", "rm", "zn", "lstat"]
    fig_scatter, ax_scatter = plt.subplots(figsize=(12, 8))
    scatter_matrix(train_set[attributes], ax=ax_scatter)
    st.pyplot(fig_scatter)

# --- 6. Model Training & Prediction ---
st.header("6. Model Training & Prediction")
st.markdown("Adjust the feature values below to predict the median house value (`medv`).")

# Prepare X and y
X = df.drop(columns=['medv'])
y = df['medv']

# Cache the model so it doesn't retrain on every slider movement
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Show basic model performance
test_preds = model.predict(test_set.drop(columns=['medv']))
r2 = r2_score(test_set['medv'], test_preds)
st.info(f"**Model Performance:** Random Forest Regressor R² Score on the Test Set: **{r2:.3f}**")

st.subheader("Input Feature Values")

# Generate input fields dynamically based on the dataset's min, max, and mean
input_data = {}
cols = st.columns(3) # Organize inputs into 3 columns

for i, col_name in enumerate(X.columns):
    col_idx = i % 3
    with cols[col_idx]:
        min_val = float(X[col_name].min())
        max_val = float(X[col_name].max())
        mean_val = float(X[col_name].mean())
        
        # Check if the feature is a binary/categorical variable like 'chas'
        if len(X[col_name].unique()) <= 2:
            input_data[col_name] = st.selectbox(f"{col_name}", options=sorted(X[col_name].unique()))
        else:
            # For continuous variables, use sliders
            input_data[col_name] = st.slider(f"{col_name}", min_value=min_val, max_value=max_val, value=mean_val, format="%.2f")

# Convert user input into a DataFrame
input_df = pd.DataFrame([input_data])

# Predict Button
if st.button("Predict House Price", type="primary"):
    prediction = model.predict(input_df)[0]
    
    # In the Boston dataset, 'medv' is measured in $1000s
    predicted_price = prediction * 1000
    
    st.success(f"### Estimated Median House Value: ${predicted_price:,.2f}")