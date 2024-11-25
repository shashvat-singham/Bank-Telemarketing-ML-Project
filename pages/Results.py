import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Function to handle logout
def logout():
    if 'user' in st.session_state:
        del st.session_state['user']  # Remove user session
    st.session_state['redirect_to_auth'] = True  # Set a flag for redirect
    st.rerun()  # Refresh to apply changes

# Check for redirect to authentication
if 'redirect_to_auth' in st.session_state and st.session_state['redirect_to_auth']:
    st.session_state.clear()  # Clear the session state
    st.session_state['redirect'] = True  # Set redirect flag
    st.rerun()  # Redirect to authentication

# Check if user is logged in
if 'user' not in st.session_state:
    st.warning("Please log in to access this page.")
    st.stop()  # Stop execution if not logged in

warnings.filterwarnings('ignore')

# Add Logout button at the top
with st.sidebar:  # You can use the sidebar for navigation or place it elsewhere
    if st.button("Logout"):
        logout()

