import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("sales_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Get user input
input_data = {
    'Item_Weight': float(input("Enter Item Weight (e.g., 12.5): ")),
    'Item_Fat_Content': input("Enter Fat Content (Low Fat / Regular): "),
    'Item_Visibility': float(input("Enter Item Visibility (e.g., 0.05): ")),
    'Item_Type': input("Enter Item Type (e.g., Fruits and Vegetables): "),
    'Item_MRP': float(input("Enter Item MRP (e.g., 200.0): ")),
    'Outlet_Identifier': input("Enter Outlet Identifier (e.g., OUT049): "),
    'Outlet_Establishment_Year': int(input("Enter Outlet Establishment Year (e.g., 1999): ")),
    'Outlet_Size': input("Enter Outlet Size (Small / Medium / High): "),
    'Outlet_Location_Type': input("Enter Outlet Location Type (e.g., Tier 1): "),
    'Outlet_Type': input("Enter Outlet Type (e.g., Supermarket Type1): ")
}

# Apply label encoding to categorical features
for col, encoder in encoders.items():
    if col in input_data:
        try:
            input_data[col] = encoder.transform([input_data[col]])[0]
        except ValueError:
            print(f"Invalid input for '{col}'. Please enter a value from the training data.")
            exit()

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict sales
predicted_sales = model.predict(input_df)[0]
print(f"\n✅ Predicted Sales: ₹{predicted_sales:.2f}")
