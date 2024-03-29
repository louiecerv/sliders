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

    st.title('Streamlit App')
    st.text('This app demonstrates how to update the values from the sliders.')
    if "n_samples" not in st.session_state:        
            st.session_state['n_samples'] = 0

    if "random_state" not in st.session_state:        
            st.session_state['random_state'] = 0


    if "n_clusters" not in st.session_state:        
            st.session_state['n_clusters'] = 0

    # Create a slider with a label and initial value

    
    n_samples = st.slider(
        label="Number of samples (200 to 4000):",
        min_value=200,
        max_value=4000,
        step=200,
        value=1000,  # Initial value
        on_change=update_values(),
        key="n_samples"
    )

   
    random_state = st.slider(
        label="Random seed (between 0 and 100):",
        min_value=0,
        max_value=100,
        value=42,  # Initial value
        on_change=update_values(),
        key="random_state"
    )
   

    n_clusters = st.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
        on_change=update_values(),
        key ="n_clusters"
    )

    # Call the function once to display initial values
    # update_values()

import streamlit as st

def update_values():
  """Prints the values of the three sliders."""
  n_samples = st.session_state['n_samples']
  random_state = st.session_state['random_state']
  n_clusters = st.session_state['n_clusters']

  dataarr = [[n_samples, random_state, n_clusters]]
  st.write("The values from the sliders are: " + str(dataarr))

#run the app
if __name__ == "__main__":
    app()
