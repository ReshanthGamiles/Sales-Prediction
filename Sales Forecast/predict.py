import pandas as pd
import joblib
import gradio as gr

# Load model and encoders
model = joblib.load("sales_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# 🧠 Reverse mapping for display dropdowns (optional)
fat_content_options = ['Low Fat', 'Regular']
outlet_size_options = ['Small', 'Medium', 'High']
location_type_options = ['Tier 1', 'Tier 2', 'Tier 3']

# Main prediction function
def predict_sales(
    Item_Weight,
    Item_Fat_Content,
    Item_Visibility,
    Item_Type,
    Item_MRP,
    Outlet_Identifier,
    Outlet_Establishment_Year,
    Outlet_Size,
    Outlet_Location_Type,
    Outlet_Type
):
    # ✅ Clean input similar to training phase
    clean_fat = Item_Fat_Content.strip().lower()
    if clean_fat in ['low fat', 'lf', 'lowfat']:
        Item_Fat_Content = 'Low Fat'
    elif clean_fat in ['regular', 'reg']:
        Item_Fat_Content = 'Regular'
    else:
        return "❌ Invalid Fat Content. Use 'Low Fat' or 'Regular'."

    # Prepare input dictionary
    input_data = {
        'Item_Weight': Item_Weight,
        'Item_Fat_Content': Item_Fat_Content,
        'Item_Visibility': Item_Visibility,
        'Item_Type': Item_Type,
        'Item_MRP': Item_MRP,
        'Outlet_Identifier': Outlet_Identifier,
        'Outlet_Establishment_Year': Outlet_Establishment_Year,
        'Outlet_Size': Outlet_Size,
        'Outlet_Location_Type': Outlet_Location_Type,
        'Outlet_Type': Outlet_Type
    }

    # Apply label encoding
    for col, encoder in encoders.items():
        if col in input_data:
            try:
                input_data[col] = encoder.transform([input_data[col]])[0]
            except ValueError:
                return f"❌ Invalid value for '{col}'. Please use a value seen in training."

    # Predict
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return f"✅ Predicted Sales: ₹{prediction:.2f}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Item Weight", value=12.5),
        gr.Dropdown(choices=fat_content_options, label="Item Fat Content", value="Low Fat"),
        gr.Number(label="Item Visibility", value=0.05),
        gr.Textbox(label="Item Type", value="Fruits and Vegetables"),
        gr.Number(label="Item MRP", value=200.0),
        gr.Textbox(label="Outlet Identifier", value="OUT049"),
        gr.Number(label="Outlet Establishment Year", value=1999),
        gr.Dropdown(choices=outlet_size_options, label="Outlet Size", value="Medium"),
        gr.Dropdown(choices=location_type_options, label="Outlet Location Type", value="Tier 1"),
        gr.Textbox(label="Outlet Type", value="Supermarket Type1")
    ],
    outputs=gr.Text(label="Prediction Output"),
    title="🛒 Sales Forecast App",
    description="Fill in the product & outlet details to predict Item Outlet Sales."
)

# Launch app
demo.launch()
