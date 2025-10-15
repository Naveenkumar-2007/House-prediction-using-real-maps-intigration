import numpy as np
import pandas as pd
import os,sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from Exception import CustomException
@dataclass
class DATA_INGESTION_CONFIG:
    raw_data_path:str=os.path.join("artifects","raw.csv")
    train_data_path:str=os.path.join("artifects","train.csv")
    test_data_path:str=os.path.join("artifects","test.csv")

class DATA_INGESTION:
    def __init__(self):
        self.data_ingestion=DATA_INGESTION_CONFIG

    def data_intiation(self):
        try:
            df=pd.read_csv("C:\\Users\\navee\\Cisco Packet Tracer 8.2.2\\saves\\india_house\\notebook\\indian_housing_data.csv")
            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False,header=True)
            train_path,test_path=train_test_split(df,test_size=0.2,random_state=42)
            train_path.to_csv(self.data_ingestion.train_data_path,index=False,header=True)
            test_path.to_csv(self.data_ingestion.test_data_path,index=False,header=True)
            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )
        except Exception as ex:
            raise CustomException(ex,sys)
if __name__=="__main__":
    obj=DATA_INGESTION()
    train_data,test_data=obj.data_intiation()
        