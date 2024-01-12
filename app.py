from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingetion import DataIngetion,DataIngetionConfig





if __name__=="__main__":
    logging.info("The execution has started")

    try:
        
        data_ingetion=DataIngetion()
        data_ingetion.initiate_data_ingetion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)


