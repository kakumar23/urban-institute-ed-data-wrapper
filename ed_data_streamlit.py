import numpy as np
import pandas as pd
import requests
import json
import os
import dotenv
import sys
import nltk
from nltk.tokenize import word_tokenize
import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
import html
from sentence_transformers import SentenceTransformer, util
import ast
import re
from typing import Dict, Optional, List
import streamlit as st
from urban_institute_wrapper_prototype import VariableExtractor as ve
from urban_institute_wrapper_prototype import query_endpoint_metadata as qem

# Initialize the extractor (do this once)
varlist_url = "https://educationdata.urban.org/api/v1/api-endpoint-varlist/"
extractor = ve(varlist_url)

# Example endpoint dictionary — you’ll likely have this from your endpoint matcher logic
example_endpoint = {
    "name": "CCD - Enrollment",
    "url_template": "/api/v1/schools/ccd/enrollment/{year}/{grade}/race/"
}

# Set page title, inputs
st.title("Education Data Natural Language Interface")
query = st.text_input("Enter your query: ")





if query:
    try:
        # Generate metadata, find optimal endpoint
        metadata = qem(query)
        metadata.procces_endpoints()
        best_endpoint, score = metadata.find_best_endpoint()

        st.json(best_endpoint)

        # Extract variables
        extracted = extractor.extract_variables(query, best_endpoint["url"])

        # Fill in URL
        final_url = extractor.fill_url(best_endpoint["url"], extracted)
        
        st.markdown(f"**Matched URL:** `{final_url}`")

        # Request Data
        resp = requests.get(final_url)
        resp.raise_for_status()

        data = resp.json()
        # Convert results to a DataFrame (if available)
        results = data.get("results", [])
        if results:
            st.success("Data fetched successfully!")
            df = pd.DataFrame(results)
            st.write("### Sample Data (First 5 Rows):")
            st.dataframe(df.head())  # Display first 5 rows
        else:
            st.warning("No results found in the response.")

        

        

    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An error occurred: {e}")