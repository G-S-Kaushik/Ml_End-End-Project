import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.DataTransformationConfig=DataTransformationConfig()

    def get_data_tranformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            df = pd.read_csv('notebook/data/raw.csv')
            X = df.drop(columns=['math_score'],axis=1)
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            """
            Converting nan and outlire values to median value and standarizing it
            """
            num_pipline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("Scaler",StandardScaler())
            ])
            
            """
            Converting nan and outlire values to mode value and standarizing it
            """
            cat_pipline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("OneHotEncoder",OneHotEncoder()),
                ("Scaler",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{cat_features}")
            logging.info(f"Numerical Columns:{num_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipline,num_features),
                    ("cat_pipline",cat_pipline,cat_features)
                ]
            )
            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading Train and test file")

            preprocessing_obj=self.get_data_tranformer_object()

            target_column_name="math_score"

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test data set to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on traing and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.DataTransformationConfig.preprocessor_obj_file_path ,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.DataTransformationConfig.preprocessor_obj_file_path
            )



        except Exception as e :
            raise CustomException(e,sys)