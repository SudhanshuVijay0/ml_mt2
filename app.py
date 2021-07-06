import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("mt2model.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('mt2data.csv')
X = dataset.iloc[:,0:9].values


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:8])
#Replacing missing data with the calculated mean value  
X[:, 1:8]= imputer.transform(X[:, 1:8])  


def predict_note_authentication(meanfreq, sd, median, IQR, skew, kurt, mode, centroid, dfrange):
  output= model.predict(sc.transform([[meanfreq, sd, median, IQR, skew, kurt, mode, centroid, dfrange]]))
  print("Model has predicted ",output)
  if output==[0]:
    prediction="The sample is of male voice...!"
   

  if output==[1]:
    prediction="The sample is of female voice...!"
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:black;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Experiment Deployment By Sudhanshu Vijay</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Prediction for voice sample")
    meanfreq = st.number_input('Insert meanfreq ')
    sd = st.number_input('Inset sd')
    median = st.number_input('Inset median')
    IQR = st.number_input('Inset IQR')
    skew = st.number_input('Inset skew')
    kurt = st.number_input('Inset kurt')
    mode = st.number_input('Inset mode')
    centroid = st.number_input('Insert centroid')
    dfrange = st.number_input('Insert dfrange')
    
    result=""
    if st.button("Predict"):
      result=predict_note_authentication(meanfreq, sd, median, IQR, skew, kurt, mode, centroid, dfrange)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Sudhanshu Vijay")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
