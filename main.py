from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load the trained model 
try:
    model = joblib.load('modal/car_price_prediction_model.pkl')
except Exception as e:
    print(f"Error loading the model: {e}")

# Initialize FastAPI
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for the input data
class CarData(BaseModel):
    year: int
    present_price: float
    kms_driven: int
    fuel_type: int
    seller_type: int
    transmission: int
    owner: int

@app.get("/", response_class=HTMLResponse)
async def serve_index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_car_price(car_data: CarData):
    # Convert the input data into a pandas DataFrame
    print(car_data)
    input_data = pd.DataFrame([{
        'Year': car_data.year,
        'Present_Price': car_data.present_price,
        'Kms_Driven': car_data.kms_driven,
        'Fuel_Type': car_data.fuel_type,
        'Seller_Type': car_data.seller_type,
        'Transmission': car_data.transmission,
        'Owner': car_data.owner
    }])

    # Make the prediction using the trained model
    predicted_price = model.predict(input_data)

    # Return the predicted price as a response
    return {"predicted_price": predicted_price[0]}
