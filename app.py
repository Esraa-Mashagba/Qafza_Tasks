from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd


# Load the pre-trained model
with open(r"C:\Users\e.mashagba.ext\Desktop\Qafza_Training\House_Price_prediction\pred.pkl", 'rb') as f:
    model = pickle.load(f)



app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Define the data schema with all features
class HouseData(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: int
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: int
    BsmtFinType2: str
    BsmtUnfSF: int
    TotalBsmtSF: int
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    GrLivArea: int
    BsmtFullBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: int
    GarageFinish: str
    GarageCars: int
    GarageArea: int
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: str
    Fence: str
    MiscFeature: str
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

@app.post("/predict")
def predict(data: HouseData):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    try:
        # Convert categorical variables into one-hot encoding
        input_df = pd.get_dummies(input_df)
        
        # Ensure the input data columns match the model's expected feature columns
        model_columns = model.feature_names_in_  # or a saved list of expected columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Convert to numpy array for prediction
        input_data = input_df.values
    except Exception as e:
        return {"error": f"Encoding failed: {e}"}
    
    # Make the prediction
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}


