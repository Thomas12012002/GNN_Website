import streamlit as st
import pandas as pd
import torch
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Streamlit app
st.title("Housing Price Forecast with GNN")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Normalize numerical attributes
    scaler = MinMaxScaler()
    normalized_df = df.copy()
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        normalized_df[col] = scaler.fit_transform(df[[col]])

    # Attribute ranking
    st.write("Rank Attributes:")
    rankings = {}
    for col in normalized_df.columns:
        rank = st.slider(f"Rank for {col}", min_value=1, max_value=len(normalized_df.columns), value=1)
        rankings[col] = rank

    ranked_columns = sorted(rankings, key=rankings.get)
    ranked_df = normalized_df[ranked_columns]

    # Estimate coefficients for utility function
    synthetic_coefficients = {col: idx / len(ranked_columns) for idx, col in enumerate(ranked_columns)}
    real_coefficients = {col: synthetic_coefficients[col] * 0.8 for col in synthetic_coefficients}

    st.write("Synthetic Coefficients:", synthetic_coefficients)
    st.write("Real Coefficients:", real_coefficients)

    # Prepare graph data for GNN
    num_features = len(ranked_columns)
    x = torch.tensor(ranked_df.values, dtype=torch.float)
    edge_index = torch.tensor([[i, j] for i in range(num_features) for j in range(num_features)], dtype=torch.long).t()

    # Define target variable
    y = torch.tensor(df['Price'].values, dtype=torch.float) if 'Price' in df.columns else torch.zeros(len(df))

    # Train GNN Model
    model = GNNModel(num_features=num_features, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = loss_fn(out.flatten(), y)
        loss.backward()
        optimizer.step()

    # Predict optimal subsets
    with torch.no_grad():
        model.eval()
        predictions = model(x, edge_index).flatten()
        top_k_indices = predictions.argsort(descending=True)[:5]
        optimal_subsets = df.iloc[top_k_indices.tolist()]

    st.write("Optimal Subsets:")
    st.write(optimal_subsets)
