import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Correlation Heatmap Generator",
    page_icon="📊",
    layout="wide"
)

# Set the title for the app
st.title("Data Correlation Heatmap Generator")
st.write("Upload your data to visualize correlations between variables.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display dataset info
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Information")
        buffer = st.empty()
        with buffer.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Number of rows: {df.shape[0]}")
                st.write(f"Number of columns: {df.shape[1]}")
            with col2:
                st.write(f"Missing values: {df.isna().sum().sum()}")
                st.write(f"Duplicate rows: {df.duplicated().sum()}")
        
        # Data preprocessing options
        st.subheader("Preprocessing Options")
        
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("The dataset must contain at least 2 numeric columns to create a correlation heatmap.")
        else:
            # Allow users to select columns
            selected_cols = st.multiselect(
                "Select columns for correlation analysis (numeric only):",
                options=numeric_cols,
                default=numeric_cols
            )
            
            # Handle missing values
            missing_handling = st.radio(
                "Handle missing values by:",
                options=["Drop rows with any missing values", "Fill missing values with mean"],
                index=0
            )
            
            if len(selected_cols) < 2:
                st.warning("Please select at least 2 columns for correlation analysis.")
            else:
                # Process the data
                df_selected = df[selected_cols].copy()
                
                if missing_handling == "Drop rows with any missing values":
                    df_selected = df_selected.dropna()
                else:
                    df_selected = df_selected.fillna(df_selected.mean())
                
                # Calculate correlation matrix
                corr_matrix = df_selected.corr()
                
                # Display correlation matrix
                st.subheader("Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))
                
                # Create heatmap
                st.subheader("Correlation Heatmap")
                
                # Heatmap customization options
                col1, col2 = st.columns(2)
                with col1:
                    heatmap_color = st.selectbox(
                        "Color palette:",
                        options=["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds"],
                        index=0
                    )
                    show_values = st.checkbox("Show correlation values", value=True)
                
                with col2:
                    fig_size = st.slider("Figure size:", min_value=5, max_value=20, value=10)
                    text_size = st.slider("Text size:", min_value=6, max_value=14, value=8)
                
                # Generate heatmap
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Create heatmap
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    cmap=heatmap_color,
                    annot=show_values,
                    fmt=".2f",
                    linewidths=.5,
                    square=True,
                    cbar_kws={"shrink": .8},
                    annot_kws={"size": text_size}
                )
                
                plt.title("Correlation Heatmap", fontsize=16)
                st.pyplot(fig)
                
                # Add download button for the heatmap
                st.subheader("Download Options")
                
                # Save plot to a BytesIO object
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                
                # Provide download button
                st.download_button(
                    label="Download Heatmap as PNG",
                    data=buf,
                    file_name="correlation_heatmap.png",
                    mime="image/png"
                )
                
                # Download correlation matrix as CSV
                csv = corr_matrix.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Correlation Matrix as CSV",
                    data=csv,
                    file_name='correlation_matrix.csv',
                    mime='text/csv',
                )
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    # Show example instructions when no file is uploaded
    st.info("Please upload a CSV file to generate a correlation heatmap.")
    
    # Add some helpful information
    st.markdown("""
    ### How to use this tool:
    1. Upload a CSV file containing your dataset
    2. Select the numeric columns you want to include in the correlation analysis
    3. Choose how to handle missing values
    4. Customize the appearance of the heatmap
    5. Download the heatmap or correlation matrix for use in your reports
    
    The correlation coefficient ranges from -1 to 1, where:
    - 1: Perfect positive correlation
    - 0: No correlation
    - -1: Perfect negative correlation
    """)

# Add footer with additional information
st.markdown("---")
st.markdown("*Note: Only numeric columns can be used for correlation analysis.*")