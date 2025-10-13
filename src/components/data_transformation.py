import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import joblib
from src.components.data_ingestion import DATA_INGESTION
@dataclass
class DATATRANSFORMATION:
    preprocessing:str=os.path.join("artifects","preprocessing.pkl")
    label_encoder:str=os.path.join("artifects","labelencoding.pkl")
class data_transformation:
    def __init__(self):
        self.preprocess=DATATRANSFORMATION()
        self.label_encoders={}
    def data_transformation(self, input_df):
        try:
            # Get all column names from the dataframe (after encoding)
            all_columns = input_df.columns.tolist()
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            preprocessing=ColumnTransformer(
                transformers=[
                    ("nums",num_pipeline,all_columns)
                ]

            )
            return preprocessing
        except Exception as ex:
            print("error",ex,sys)
            raise ex
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features using Label Encoding
        """
        try:
            df_copy = df.copy()
            
            categorical_columns = [
                'State', 'City', 'Locality', 'Property_Type',
                'Furnished_Status', 'Public_Transport_Accessibility',
                'Parking_Space', 'Security', 'Facing', 'Owner_Type',
                'Availability_Status'
            ]
            
            # Also encode Amenities (which is a string with multiple values)
            if 'Amenities' in df_copy.columns:
                # Count number of amenities
                df_copy['Amenities_Count'] = df_copy['Amenities'].apply(
                    lambda x: len(str(x).split(',')) if pd.notna(x) else 0
                )
                # Create binary features for common amenities
                df_copy['Has_Gym'] = df_copy['Amenities'].str.contains('Gym', case=False, na=False).astype(int)
                df_copy['Has_Pool'] = df_copy['Amenities'].str.contains('Pool', case=False, na=False).astype(int)
                df_copy['Has_Playground'] = df_copy['Amenities'].str.contains('Playground', case=False, na=False).astype(int)
                df_copy['Has_Garden'] = df_copy['Amenities'].str.contains('Garden', case=False, na=False).astype(int)
                df_copy['Has_Clubhouse'] = df_copy['Amenities'].str.contains('Clubhouse', case=False, na=False).astype(int)
                
                # Drop original Amenities column
                df_copy = df_copy.drop('Amenities', axis=1)
            
            for col in categorical_columns:
                if col in df_copy.columns:
                    if is_training:
                        le = LabelEncoder()
                        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Handle unseen categories
                            df_copy[col] = df_copy[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
            
            return df_copy
            
        except Exception as e:
            print(f"Error encoding categorical features: {str(e)}")
            raise e
        
    def data_intiate_trans(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            if "ID" in train_df.columns:
                train_df=train_df.drop("ID",axis=1)
            if "ID" in test_df.columns:
                test_df=test_df.drop("ID",axis=1)
            
            Traget_feature="Price_in_Lakhs"
            input_train_df=train_df.drop(columns=[Traget_feature],axis=1)
            target_train_df=train_df[Traget_feature]

            input_test_df=test_df.drop(columns=[Traget_feature],axis=1)
            target_test_df=test_df[Traget_feature]
            ##update Encoding columns
            input_train_df=self.encode_categorical_features(input_train_df,is_training=True)
            input_test_df=self.encode_categorical_features(input_test_df,is_training=False)
            
            #  Check data types
            print("Columns after encoding:")
            print(input_train_df.dtypes)
            print("\nNon-numeric columns:")
            print(input_train_df.select_dtypes(include=['object']).columns.tolist())
            
            #preprocess
            preprocessing=self.data_transformation(input_train_df)
            input_pre_trainer=preprocessing.fit_transform(input_train_df)
            input_pre_testing=preprocessing.transform(input_test_df)

            train_arr=np.c_[
                input_pre_trainer,np.array(target_train_df)
            ]
            test_arr=np.c_[
                input_pre_testing,np.array(target_test_df)
            ]

            os.makedirs(os.path.dirname(self.preprocess.preprocessing),exist_ok=True)
            joblib.dump(preprocessing,self.preprocess.preprocessing)
            joblib.dump(self.label_encoders,self.preprocess.label_encoder)
            return(
                train_arr,
                test_arr,
                self.preprocess.preprocessing
            )
        except Exception as e:
            print(f"Error : {str(e)}")
            raise e
        
if __name__=="__main__":
    obj=DATA_INGESTION()
    train_data,test_data=obj.data_intiation()
    data_transform=data_transformation()
    train_arr,test_arr,_=data_transform.data_intiate_trans(train_data,test_data)