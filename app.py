from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingetion import DataIngetion,DataIngetionConfig
from src.mlproject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig




if __name__=="__main__":
    logging.info("The execution has started")

    try:
        
        data_ingetion=DataIngetion()
        train_data_path,test_data_path=data_ingetion.initiate_data_ingetion()

        data_tranformation=DataTransformation()
        train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

        ##Model Training
        model_trainer=ModelTrainer()
        print(model_trainer.initate_model_trainer(train_arr,test_arr))
    
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)


