#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    st.title('Logistic Regression, Naive Bayes Classifiers and Support Vector Machine')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
 
    # Create a slider with a label and initial value
    st.session_state['n_samples']) = st.slider(
        label="Number of samples (200 to 4000):",
        min_value=200,
        max_value=4000,
        step=200,
        value=1000,  # Initial value
        on_change=update_values(st.session_state['n_samples']), 
        key="n_samples"
    )
  
    st.session_state['random_state'] = st.slider(
        label="Random seed (between 0 and 100):",
        min_value=0,
        max_value=100,
        value=42,  # Initial value
        on_change=update_values(st.session_state['random_state']), 
        key="random_state"
    )
   
    st.session_state['n_clusters'] = st.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
        on_change=update_values(st.session_state['n_clusters']), 
        key="n_clusters"

    )

    # Call the function once to display initial values
    # update_values()

import streamlit as st

def update_values(key):
  """Prints the values of the three sliders."""
  st.write("N Samples:" + str(key))

#run the app
if __name__ == "__main__":
    app()
