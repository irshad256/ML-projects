import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('/home/user/Downloads/train-data.csv',index_col=0)
df.drop(['New_Price'],inplace=True,axis=1)
df=df.dropna()
df[['Mileage','mileage_unit']]=df.Mileage.str.split(expand=True)
df.drop(['mileage_unit'],axis=1,inplace=True)
df.Mileage=df.Mileage.astype(float)
df[['Engine','unit']]=df.Engine.str.split(expand=True)
df.Engine=df.Engine.astype(float)
df.drop(['unit'],inplace=True,axis=1)
df[['Power','unit']]=df.Power.str.split(expand=True)
i=df[df.Power=='null'].index
df.drop(i,inplace=True)
df.Power=df.Power.astype(float)
df.drop(['unit'],inplace=True,axis=1)
df['Name']=df['Name'].str.upper()



car_name=df['Name'].unique()
car_location=df['Location'].unique()
car_year=df['Year'].unique()
car_year.sort()
fuel_type=df['Fuel_Type'].unique()
transmission=df['Transmission'].unique()
owner_type=df['Owner_Type'].unique()


st.title("Used Cars Price Prediction")
car_Name = st.selectbox("Name",car_name)
car_Location=st.selectbox('Location',car_location)
car_Year=st.selectbox('Year',car_year)
car_Kilometers=st.number_input('Kilometers')
car_Fuel_Type=st.selectbox('Fuel Type',fuel_type)
car_Transmission=st.selectbox('Transmission',transmission)
car_Owner_Type=st.selectbox("Owner_Type",owner_type)
car_Mileage=st.number_input('Mileage')
car_Engine=st.number_input('Engine')
car_Power=st.number_input('Power')
car_Seats=st.number_input('Seats')
df.loc[10000]=[car_Name,car_Location,car_Year,car_Kilometers,car_Fuel_Type,car_Transmission,car_Owner_Type,car_Mileage,car_Engine,car_Power,car_Seats,0]

le=LabelEncoder()
df.Name=le.fit_transform(df.Name)
df.Location=le.fit_transform(df.Location)
df=pd.get_dummies(df,drop_first=True)
df.drop(['Price'],axis=1,inplace=True)
p=df.loc[[10000]]
best_model=pickle.load(open('/home/user/Downloads/model.sav','rb'))

def fun():
    prediction = best_model.predict(p)
    st.write('Price',prediction)
    return
if st.button('Predict'):
    fun()
