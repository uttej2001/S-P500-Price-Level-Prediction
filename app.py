import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

st.title('Stock Prediction')

df1 = pd.read_csv("GOOGL_data.csv")

df2 = pd.read_csv("AAPL_data.csv")

#Describing the data

st.subheader('Google Stock Data from S&P 500')
st.write(df1.describe())

st.subheader('Apple Stock Data from S&P 500')
st.write(df2.describe())

#Visualizations
#google
st.subheader('Google: Closing Price vs Time Chart')
fig1 = plt.figure(figsize = (12,6))
plt.plot(df1.close, label = 'Google')
st.pyplot(fig1)

#apple
st.subheader('Apple: Closing Price vs Time Chart ')
fig2 = plt.figure(figsize = (12,6))
plt.plot(df2.close, label = 'Apple')
st.pyplot(fig2)

#google
st.subheader('Google: Closing Price vs Time Chart with 100MA')
ma100 = df1.close.rolling(100).mean()
fig3 = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df1.close)
st.pyplot(fig3)

#apple
st.subheader('Apple: Closing Price vs Time Chart with 100MA')
ma100_2 = df2.close.rolling(100).mean()
fig4 = plt.figure(figsize = (12,6))
plt.plot(ma100_2)
plt.plot(df2.close)
st.pyplot(fig4)

#google
st.subheader('Google: Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df1.close.rolling(100).mean()
ma200 = df1.close.rolling(200).mean()
fig5 = plt.figure(figsize = (12,6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df1.close, 'b')
st.pyplot(fig5)

#apple
st.subheader('Apple: Closing Price vs Time Chart with 100MA & 200MA')
ma100_2 = df2.close.rolling(100).mean()
ma200_2 = df2.close.rolling(200).mean()
fig6 = plt.figure(figsize = (12,6))
plt.plot(ma100_2, 'g')
plt.plot(ma200_2, 'r')
plt.plot(df2.close, 'b')
st.pyplot(fig6)

# Splitting data into training and testing
#google
data_training = pd.DataFrame(df1['close'][0:int(len(df1)*0.70)])
data_testing = pd.DataFrame(df1['close'][int(len(df1)*0.70): int(len(df1))])

#apple
data_training_2 = pd.DataFrame(df2['close'][0:int(len(df2)*0.70)])
data_testing_2 = pd.DataFrame(df2['close'][int(len(df2)*0.70): int(len(df2))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#google
data_training_array = scaler.fit_transform(data_training)

#apple
data_training_array_2 = scaler.fit_transform(data_training_2)

# Load Our Model
model = load_model('model.h5')

#Testing part

#google
past_30_days = data_training.tail(30)
final_df1 = past_30_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df1)

#apple
past_30_days_2 = data_training_2.tail(30)
final_df2 = past_30_days_2.append(data_testing_2, ignore_index=True)
input_data_2 = scaler.fit_transform(final_df2)

#google
x_test = []
y_test = []

for i in range(30, input_data.shape[0]):
    x_test.append(input_data[i-30: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

#apple
x_test_2 = []
y_test_2 = []

for i in range(30, input_data_2.shape[0]):
    x_test_2.append(input_data_2[i-30: i])
    y_test_2.append(input_data_2[i, 0])
    
x_test_2, y_test_2 = np.array(x_test_2), np.array(y_test_2)

#Prediction

#google
y_predicted = model.predict(x_test)

scaler1 = scaler.scale_
scale_factor = 1/scaler1[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#apple
y_predicted_2 = model.predict(x_test_2)

scaler2 = scaler.scale_
scale_factor_2 = 1/scaler2[0]
y_predicted_2 = y_predicted_2 * scale_factor_2
y_test_2 = y_test_2 * scale_factor_2

#Final Graph

#google
st.subheader('Google: Predictions vs Actual')
fig7 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Actual Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig7)

#apple
st.subheader('Apple: Predictions vs Actual')
fig8 = plt.figure(figsize = (12,6))
plt.plot(y_test_2, 'b', label = 'Actual Price')
plt.plot(y_predicted_2, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig8)

#Plots
st.subheader('Final: Predictions vs Actual')
fig9 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Google-Actual Price')
plt.plot(y_predicted, 'r', label = 'Google-Predicted Price')
plt.plot(y_test_2, 'cyan', label = 'Apple-Actual Price')
plt.plot(y_predicted_2, 'g', label = 'Apple-Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig9)