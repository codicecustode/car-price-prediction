from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load the trained model and scaler
model = joblib.load('car_price_prediction_model.pkl')
#scaler = joblib.load('scaler.pkl')  # Use a scaler if you applied scaling during training

# Initialize FastAPI
app = FastAPI()

# Allow all origins for testing (in production, you can limit the allowed origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. You can specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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

    print(input_data)

    # Apply scaling (if needed)
    #input_data_scaled = scaler.transform(input_data)  # Apply the scaler to the input data if scaling was used during training

    # Make the prediction using the trained model
    predicted_price = model.predict(input_data)

    # Return the predicted price as a response
    return {"predicted_price": predicted_price[0]}
