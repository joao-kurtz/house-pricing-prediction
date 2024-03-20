from importAndProcessing import importAndProcessing
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def train_and_evaluate_model():
    x, y = importAndProcessing()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    reg_model = LinearRegression()
    reg_model.fit(x_train, y_train)

    reg_train_predict = reg_model.predict(x_train)
    reg_test_predict = reg_model.predict(x_test)

    # save model
    model_file_path = 'savedModels\linear_regression_model.pkl'
    joblib.dump(reg_model, model_file_path)

    # evaluation metrics
    mse_train = mean_squared_error(y_train, reg_train_predict)
    mse_test = mean_squared_error(y_test, reg_test_predict)
    mae_train = mean_absolute_error(y_train, reg_train_predict)
    mae_test = mean_absolute_error(y_test, reg_test_predict)
    mape_train = mean_absolute_percentage_error(y_train, reg_train_predict) * 100
    mape_test = mean_absolute_percentage_error(y_test, reg_test_predict) * 100
    r2_train = reg_model.score(x_train, y_train)
    r2_test = reg_model.score(x_test, y_test)

    

    # Organizar métricas em um DataFrame
    evaluation_metrics = {
        "Target": ['Train', 'Test'],
        "MSE": [mse_train, mse_test],
        "MAE": [mae_train, mae_test],
        "MAPE": [mape_train, mape_test], 
        "R2 Score": [r2_train, r2_test]
    }
    evaluation_df = pd.DataFrame(evaluation_metrics)

    return evaluation_df

evaluation_df = train_and_evaluate_model()
print("model saved on savedModels folder")
print(evaluation_df)