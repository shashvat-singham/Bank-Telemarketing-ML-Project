import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import StringIO
from PIL import Image

# Function to handle logout
def logout():
    if 'user' in st.session_state:
        del st.session_state['user']
    st.session_state['redirect_to_auth'] = True
    st.rerun()

# Check for redirect to authentication
if 'redirect_to_auth' in st.session_state and st.session_state['redirect_to_auth']:
    st.session_state.clear()
    st.session_state['redirect'] = True
    st.rerun()

# Check if user is logged in
if 'user' not in st.session_state:
    st.warning("Please log in to access this page.")
    st.stop()

warnings.filterwarnings('ignore')

# Add Logout button
with st.sidebar:
    if st.button("Logout"):
        logout()

# Title of the app
st.title("Bank Telemarketing Campaign Optimization")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload the bank dataset CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    dataset = pd.read_csv(uploaded_file, sep=';')
    
    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    # Display dataset information
    st.subheader("Dataset Information")
    buffer = StringIO()
    dataset.info(buf=buffer)
    st.text(buffer.getvalue())

    # Data statistics
    st.subheader("Dataset Statistics")
    st.write(dataset.describe())

    # Data cleaning and processing
    st.subheader("Data Cleaning and Processing")
    dataset['education'] = dataset['education'].replace({
        'basic.9y': 'Basic',
        'basic.6y': 'Basic',
        'basic.4y': 'Basic'
    })
    st.write("Education levels combined as 'Basic' for consistency.")

    # Allow user to filter data
    st.subheader("Data Filtering")
    job_filter = st.selectbox("Select a job type to filter", options=dataset['job'].unique())
    filtered_data = dataset[dataset['job'] == job_filter]
    st.write(f"Data filtered by job: {job_filter}")
    st.dataframe(filtered_data.head())

    # Visualization: Job distribution
    st.subheader("Visualization: Job Distribution")
    fig1, ax1 = plt.subplots()
    dataset['job'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title("Job Distribution")
    ax1.set_xlabel("Job")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Visualization: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    sns.heatmap(numeric_dataset.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Pie chart: Education distribution
    st.subheader("Visualization: Education Distribution")
    fig3, ax3 = plt.subplots()
    dataset['education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax3, colors=sns.color_palette('pastel'))
    ax3.set_title("Education Distribution")
    ax3.set_ylabel("")
    st.pyplot(fig3)

    # Histogram: Age distribution
    st.subheader("Visualization: Age Distribution")
    fig4, ax4 = plt.subplots()
    dataset['age'].plot.hist(bins=20, ax=ax4, color='orange', edgecolor='black')
    ax4.set_title("Age Distribution")
    ax4.set_xlabel("Age")
    st.pyplot(fig4)

    # Boxplot: Age vs Balance
    st.subheader("Visualization: Age vs Balance")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=dataset, x='job', y='balance', ax=ax5, palette='Set2')
    ax5.set_title("Balance Distribution by Job")
    ax5.set_xlabel("Job")
    ax5.set_ylabel("Balance")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

    # Line Chart: Campaign Outcome Over Time
    st.subheader("Visualization: Campaign Outcome Over Time")
    if 'month' in dataset.columns and 'y' in dataset.columns:
        monthly_outcome = dataset.groupby('month')['y'].value_counts().unstack().fillna(0)
        fig6, ax6 = plt.subplots()
        monthly_outcome.plot(kind='line', ax=ax6)
        ax6.set_title("Campaign Outcomes by Month")
        ax6.set_xlabel("Month")
        ax6.set_ylabel("Count")
        st.pyplot(fig6)
    else:
        st.warning("Month or Campaign Outcome column missing for this chart.")

    # Pairplot: Visualizing Relationships
    st.subheader("Visualization: Pairplot")
    if len(numeric_dataset.columns) >= 2:
        pairplot_fig = sns.pairplot(numeric_dataset)
        st.pyplot(pairplot_fig)
    else:
        st.warning("Not enough numeric columns for pairplot.")
    
    # Visualization: Subscription vs Contact Age
    st.subheader("Visualization: Subscription vs Contact Age")
    fig7, ax7 = plt.subplots()
    sns.boxplot(data=dataset, x='y', y='age', ax=ax7, palette='coolwarm')
    ax7.set_title("Subscription vs Contact Age")
    ax7.set_xlabel("Subscribed (Yes/No)")
    ax7.set_ylabel("Age")
    st.pyplot(fig7)

    # Visualization: Subscribed a Term Deposit vs Contact Rate by Month
    st.subheader("Visualization: Subscribed Term Deposit vs. Contact Rate by Month")
    if 'month' in dataset.columns and 'y' in dataset.columns:
     month_subscription = dataset.groupby('month')['y'].value_counts(normalize=True).unstack().fillna(0) * 100
     fig8, ax8 = plt.subplots()
     month_subscription.plot(kind='bar', stacked=True, ax=ax8, colormap='viridis')
     ax8.set_title("Subscription Rate by Month")
     ax8.set_xlabel("Month")
     ax8.set_ylabel("Percentage")
     st.pyplot(fig8)
    else:
     st.warning("Month or Campaign Outcome column missing for this chart.")

    # Visualization: Probability Density Functions
    st.subheader("Visualization: Probability Density Functions (PDFs)")
    fig9, ax9 = plt.subplots()
    sns.kdeplot(data=dataset, x='age', hue='y', fill=True, common_norm=False, palette='muted', alpha=0.6, ax=ax9)
    ax9.set_title("Age Distribution by Subscription Outcome")
    ax9.set_xlabel("Age")
    st.pyplot(fig9)


    # Interactive feature: Select columns to display
    st.subheader("Select Columns to Display")
    columns = st.multiselect("Choose columns to display", options=dataset.columns.tolist(), default=dataset.columns.tolist())
    st.dataframe(dataset[columns].head())

    # Custom query feature
    st.subheader("Run a Custom Query")
    query = st.text_input("Enter a Pandas query (e.g., 'age > 30 and job == \"admin.\"')")
    if query:
        try:
            query_result = dataset.query(query)
            st.write(query_result)
        except Exception as e:
            st.error(f"Error in query: {e}")
    # Paths to the images
    model_image_path = "assets/models.png"
    roc_image_path = "assets/roc.png"

    # Load the images
    model_image = Image.open(model_image_path)
    roc_image = Image.open(roc_image_path)

    # Display the images
    st.image(model_image, caption="Model Diagram", use_container_width=True)
    st.image(roc_image, caption="ROC Curve", use_container_width=True)
else:
    st.info("Please upload a CSV file to proceed.")
