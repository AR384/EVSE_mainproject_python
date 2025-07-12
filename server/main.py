from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks,Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import pandas as pd 
import utils.myprocess as myprocess


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 도메인 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
resultDTO = {}
jobState = {}
final_result = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ APP ]" )

@app.get('/fastapi/getstation')
async def get_station():
    stations = myprocess.getStation()
    print(stations)
    return stations

@app.get('/fastapi/{station_id}')
async def get_result(station_id:str):
    filtered = myprocess.getEvse(station_id)
    print(filtered)
    return {'results': filtered}

@app.post('/fastapi/inference')
async def get_result(data:dict):
    print('받은 데디어',data)
    print('받은 데이터',data['충전소'])
    print('받은 데이터',data['충전기'])
    pred_cls,pred_reg = myprocess.doInference(data)
    print(pred_cls)
    print(pred_reg)
    return {'message':'데이터 수신 완료'}
