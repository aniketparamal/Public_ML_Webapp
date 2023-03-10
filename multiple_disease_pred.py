#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle


# In[4]:


heart_model=pickle.load(open('heart_disease_model.pkl','rb'))
diabetes_model=pickle.load(open('diabetes_model.pkl','rb'))


# In[ ]:


with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         options=['Diabetes Prediction','Heart Disease Prediction',],
                         icons=['activity','heart'],default_index=0)
    
if(selected=='Diabetes Prediction'):
    st.title('Diabetes Prediction System')

    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.text_input('Number of Pregnancies')
    with col2:
        Glucose=st.text_input('Glucose level')
    with col3:
        BloodPressure=st.text_input('BloodPressure level')
    with col1:
        SkinThickness=st.text_input('SkinThickness value')
    with col2:
        Insulin=st.text_input('Insulin Level')
    with col3:
        BMI=st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunc=st.text_input('Diabetes Pedigree Function')
    with col2:
        Age=st.text_input('Age of person')
    
    entries=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunc,Age]
    
    
    if st.button('Diagnosis'):
        diagnosis=diabetes_model.predict([entries])
        if diagnosis[0]==1:
            st.success('The person is diabetic.')
        else:
            st.success('The person is not diabetic.')
    

if(selected=='Heart Disease Prediction'):
    st.title('Heart Disease Prediction System')

    col1,col2,col3=st.columns(3)
    with col1:
        age=st.text_input('Age of Person')
    with col2:
        sex=st.text_input('Sex:(0 for Female and 1 for Male)')
    with col3:
        cp=st.text_input('Type of chest pain:(0 or 1 or 2 or 3)')
    with col1:
        trestbps=st.text_input('Resting Blood Pressure')
    with col2:
        chol=st.text_input('Cholestrol level')
    with col3:
        fbs=st.text_input('Fasting Blood Sugar level')
    with col1:
        restecg=st.text_input('Resting electrocardiographic results:(0 or 1 or 2)')
    with col2:
        thalach=st.text_input('Maximum heart rate observed')
    with col3:
        exang=st.text_input('Exercise induced angina:(0 for No and 1 for Yes)')
    with col1:
        oldpeak=st.text_input('ST depression induced by exercise')
    with col2:
        slope=st.text_input('Slope of the peak exercise ST segment:(0,1 or 2)')
    with col3:
        ca=st.text_input('No of major vessels(0-4) colored by fluoroscopy')
    with col1:
        thal=st.text_input('Thalssemia type(0,1,2 or 3 )')
    
    entries=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    entries_arr=np.array(entries)
    entries_arr=entries_arr.reshape(1,-1)
    
    if(st.button('Diagnosis')):
        diagnosis=heart_model.predict(entries_arr)
        if diagnosis[0]==1:
            st.success("The person has heart disease")
        else:
            st.success("The person does not have any heart disease")



    



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




