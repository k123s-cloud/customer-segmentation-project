from email.mime import text
from urllib import request

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
#from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
#from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import numpy as np


from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *

import warnings
warnings.filterwarnings('ignore')

load_dotenv(override=True)
print("ENV VALUE:", os.getenv("MONGO_URL"))

#print("MONGODB_URL:", os.getenv("MONGODB_URL"))
#print("MONGODB_URL_KEY:", os.getenv("MONGODB_URL_KEY"))

app = FastAPI()



templates = Jinja2Templates(directory='templates')
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("customer.html", {"request": request})


origins = ["*"]

#app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class CustomerData(BaseModel):
    Age: int
    Education: int
    Marital_Status: int
    Parental_Status: int
    Children: int
    Income: float
    Total_Spending: float
    Days_as_Customer: int
    Recency: int
    Wines: int
    Fruits: int
    Meat: int
    Fish: int
    Sweets: int
    Gold: int
    Web: int
    Catalog: int
    Store: int
    Discount_Purchases: int
    Total_Promo: int
    NumWebVisitsMonth: int


"""class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Age : Optional[str] = None
        self.Education  : Optional[str] = None
        self.Marital_Status  : Optional[str] = None
        self.Parental_Status : Optional[str] = None
        self.Children  : Optional[str] = None
        self.Income  : Optional[str] = None
        self.Total_Spending  : Optional[str] = None
        self.Days_as_Customer  : Optional[str] = None
        self.Recency  : Optional[str] = None
        self.Wines  : Optional[str] = None
        self.Fruits  : Optional[str] = None
        self.Meat : Optional[str] = None
        self.Fish   : Optional[str] = None
        self.Sweets : Optional[str] = None
        self.Gold  : Optional[str] = None
        self.Web  : Optional[str] = None
        self.Catalog  : Optional[str] = None
        self.Store  : Optional[str] = None
        self.Discount_Purchases  : Optional[str] = None
        self.Total_Promo  : Optional[str] = None
        self.NumWebVisitsMonth  : Optional[str] = None
        

async def get_customer_data(self):
        form =  await self.request.form()
        self.Age = form.get('Age')
        self.Education = form.get('Education')
        self.Marital_Status = form.get('Marital_Status')
        self.Parental_Status = form.get('Parental_Status')
        self.Children = form.get('Children')
        self.Income = form.get('Income')
        self.Total_Spending = form.get('Total_Spending')
        self.Days_as_Customer = form.get('Days_as_Customer')
        self.Recency = form.get('Recency')
        self.Wines = form.get('Wines')
        self.Fruits = form.get('Fruits')
        self.Meat = form.get('Meat')
        self.Fish = form.get('Fish')
        self.Sweets = form.get('Sweets')
        self.Gold = form.get('Gold')
        self.Web = form.get('Web')
        self.Catalog = form.get('Catalog')
        self.Store = form.get('Store')
        self.Discount_Purchases = form.get('Discount_Purchases')
        self.Total_Promo = form.get('Total_Promo')
        self.NumWebVisitsMonth = form.get('NumWebVisitsMonth')"""

@app.get("/train")
async def trainRouteClient():
    try:
        pipeline = TrainPipeline()

        pipeline.run_pipeline()

        return JSONResponse(content={"status": True, "message": "Training successful!"})

    except Exception as e:
        import traceback
        print("TRAIN ERROR:")
        traceback.print_exc()
        
        return JSONResponse(content={"status": False, "error": str(e)},status_code=500)
    
    
@app.get("/test_env")
async def test_env():
    mongo_url = os.getenv("MONGO_DB_URL")
    return {"MONGO_URL": mongo_url }
            


"""@app.get("/test_env")
async def predictGetRouteClient(request: Request):
    try:

        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": "Rendering"},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")"""
        
    
    
@app.post("/")
async def predictRouteClient(request: Request):
    try:
        data = await request.form()
        print("Received data:", data)
        input_data = [int(data.get('Age')),
                    int(data.get('Education')),
                    int(data.get('Marital_Status')),
                    int(data.get('Parental_Status')),
                    int(data.get('Children')), 
                    int(data.get('Income')), 
                    float(data.get('Total_Spending')), 
                    int(data.get('Days_as_Customer')), 
                    int(data.get('Recency')), 
                    int(data.get('Wines')), 
                    int(data.get('Fruits')), 
                    int(data.get('Meat')), 
                    int(data.get('Fish')), 
                    int(data.get('Sweets')), 
                    int(data.get('Gold')), 
                    int(data.get('Web')), 
                    int(data.get('Catalog')), 
                    int(data.get('Store')), 
                    int(data.get('Discount_Purchases')), 
                    int(data.get('Total_Promo')), 
                    int(data.get('NumWebVisitsMonth'))]
        # ================== VALIDATION START ==================

        Age = int(data.get("Age"))
        Income = float(data.get("Income"))
        Total_Spending = float(data.get("Total_Spending"))
        Days_as_Customer = int(data.get("Days_as_Customer"))
        if Age < 18 or Age > 100:
            return templates.TemplateResponse("customer.html",{"request": request, "result": "Invalid Age ❌"})

        if Income < 0 or Income > 10000000:
            return templates.TemplateResponse("customer.html",{"request": request, "result": "Invalid Income ❌"})

        if Total_Spending < 0:
            return templates.TemplateResponse("customer.html",{"request": request, "result": "Invalid Spending ❌"})

        if Days_as_Customer < 0:
            return templates.TemplateResponse("customer.html",{"request": request, "result": "Invalid Customer Days ❌"})

        #{"request": request, "result": "Invalid Spending ❌"}

        if Days_as_Customer < 0:
            return templates.TemplateResponse( "customer.html",{"request": request, "result": "Invalid Customer Days ❌"})

# ================== VALIDATION END ==================
        
        prediction_pipeline = PredictionPipeline()
        predicted_cluster = prediction_pipeline.run_pipeline(input_data=input_data)
        #resp={"predicted_cluster": int(predicted_cluster[0])}
        cluster = int(predicted_cluster[0])
        if cluster == 0:
            label = "Basic Customer"
            color = "#f8c471"
        elif cluster == 1:
            label = "Loyal Customer"
            color = "#c3e6cb"
        elif cluster == 2:
            label = "Premium Customer"
            color = "#bee5eb"
        #return JSONResponse(content=resp)
        return templates.TemplateResponse("customer.html", {"request": request, "result": label, "bg_color": color, "text_color": "#333"})

    except Exception as e:
        print("PREDICTION ERROR:",  e)
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


@app.post("/api/predict")
async def predict_api(data: CustomerData):
    try:
        input_data = [
            data.Age,
            data.Education,
            data.Marital_Status,
            data.Parental_Status,
            data.Children,
            data.Income,
            data.Total_Spending,
            data.Days_as_Customer,
            data.Recency,
            data.Wines,
            data.Fruits,
            data.Meat,
            data.Fish,
            data.Sweets,
            data.Gold,
            data.Web,
            data.Catalog,
            data.Store,
            data.Discount_Purchases,
            data.Total_Promo,
            data.NumWebVisitsMonth
        ]

        prediction_pipeline = PredictionPipeline()
        predicted_cluster = prediction_pipeline.run_pipeline(input_data=input_data)

        return {"predicted_cluster": int(predicted_cluster[0])}

    except Exception as e:
        return JSONResponse(
            content={"status": False, "error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    
    #print("MONGODB_URL:", os.getenv("MONGODB_URL"))
    #print("MONGODB_URL_KEY:", os.getenv("MONGODB_URL_KEY"))
    
    
    uvicorn.run("app:app", host = "127.0.0.1", port =5000, reload=True)
    
