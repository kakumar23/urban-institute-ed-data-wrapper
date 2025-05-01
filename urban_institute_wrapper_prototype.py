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


# Create class for endpoint metadata extraction and query processing
class query_endpoint_metadata:
    def __init__(self, query):
        self.endpoints_url = "https://educationdata.urban.org/api/v1/api-endpoints/"
        self.response = requests.get(self.endpoints_url)
        self.endpoints = self.response.json()
        self.endpoint_list = self.endpoints.get("results", [])
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # for query and endpoint processing
        self.query = query

    # Refigure endpoint metadata for optimal NLP use
    def procces_endpoints(self):
        # initialize cleaned endpoints
        self.processed_endpoints = []

        # Extract relevant info for NLP
        for ep in self.endpoint_list:
            endpoint_id = ep.get("endpoint_id")
            source = ep.get("class_name", "")
            topic = ep.get("topic", "")
            description = ep.get("description", "")
            url = ep.get("endpoint_url", "")
            
            # Decode and parse the years
            raw_years = ep.get("years", "")
            raw_years = html.unescape(raw_years)  # Converts "&ndash;" to en dash
            try:
                start_year, end_year = map(int, raw_years.split("–"))
                years = list(range(start_year, end_year + 1))
            except ValueError:
                years = []

            # add cleaned endpoint to list
            self.processed_endpoints.append({
                "id": endpoint_id,
                "name": f"{source} - {topic}" if topic else source,
                "description": description,
                "years": years,
                "url": url
            })

    #Determine appropriate endpoint based on input query
    def find_best_endpoint(self):
        # Convert endpoint description to embeddings
        corpus = [f"{ep['name']} - {ep['description']}" for ep in self.processed_endpoints]
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)

        # Convert query to embeddings
        query_embedding = self.model.encode(self.query, convert_to_tensor=True)

        # Calculate similarity, choose best endpoint match
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        best_match = self.processed_endpoints[best_idx]
        
        return best_match, best_score



# Define class for extracting relevant variables for building the URL
class VariableExtractor:
    def __init__(self, varlist_url):
        self.root = "https://educationdata.urban.org"
        self.varlist_url = varlist_url
        self.var_metadata = self._fetch_var_metadata()
        self.reverse_mappings = self._create_reverse_mappings()
        print(f"[init] Variable patterns loaded for: {list(self.reverse_mappings.keys())}")


        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns_to_matcher()

    def _fetch_var_metadata(self):
        response = requests.get(self.varlist_url)
        response.raise_for_status()
        return response.json()

    def _parse_values(self, raw_values: str):
        decoded = html.unescape(raw_values)
        try:
            parsed = ast.literal_eval(f"{{{decoded}}}")
            return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
        except (SyntaxError, ValueError) as e:
            print(f"[parse_values] Failed to parse: {decoded[:100]}... \nError: {e}")
            return {}

    def _create_reverse_mappings(self) -> Dict[str, Dict[str, str]]:
        reverse_mappings = {}
        for var_name, metadata in self.var_metadata.items():
            if not isinstance(metadata, dict):
                continue  # skip if it's not a dictionary

            raw_values = metadata.get("values", "")
            print(raw_values)
            value_map = self._parse_values(raw_values)
            reverse_map = {}

            for code, label in value_map.items():
                label_norm = label.lower().replace("-", " ").strip()
                reverse_map[label_norm] = code

                # Add aliases
                if var_name == "grade":
                    reverse_map.update(self._generate_grade_aliases(code, label))

            reverse_mappings[var_name] = reverse_map
        return reverse_mappings

    def _generate_grade_aliases(self, code: str, label: str) -> Dict[str, str]:
        aliases = {}
        label_lower = label.lower()
        if label_lower in ["pre-k", "prekindergarten"]:
            aliases.update({"pre k": code, "pre-k": code, "prekindergarten": code})
        elif label_lower == "kindergarten":
            aliases.update({"kindergarten": code, "k": code})
        elif label_lower.isdigit():
            num = int(label_lower)
            suffix = self._ordinal_suffix(num)
            aliases.update({
                f"{num}": code,
                f"{num}th grade": code,
                f"{num}{suffix} grade": code,
                f"grade {num}": code
            })
        return aliases

    def _ordinal_suffix(self, n: int) -> str:
        return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    def _add_patterns_to_matcher(self):
        # for var_name, terms in self.reverse_mappings.items():
        #     for phrase in terms.keys():
        #         if not phrase:
        #             continue
        #         pattern = [{"LOWER": token.lower()} for token in phrase.split()]
        #         self.matcher.add(var_name, [pattern])
        for var_name, terms in self.reverse_mappings.items():
            count = 0
            for phrase in terms.keys():
                if not phrase:
                    continue
                pattern = [{"LOWER": token.lower()} for token in phrase.split()]
                self.matcher.add(var_name, [pattern])
                count += 1
            if count:
                print(f"[matcher] Added {count} patterns for variable: {var_name}")

    def get_required_vars_from_url(self, url_template: str) -> List[str]:
        return re.findall(r"\{(\w+)\}", url_template)

    def extract_variables(self, query: str, url_template: str) -> Dict[str, Optional[str]]:
        required_vars = self.get_required_vars_from_url(url_template)
        extracted = {}
        doc = self.nlp(query)

        # Extract year using NER
        if "year" in required_vars:
            for ent in doc.ents:
                if ent.label_ == "DATE" and ent.text.isdigit() and len(ent.text) == 4:
                    extracted["year"] = ent.text
                    break

        # Match other variables with spaCy matcher
        matches = self.matcher(doc)
        matched_vars = set()
        for match_id, start, end in matches:
            var_name = self.nlp.vocab.strings[match_id]
            if var_name in required_vars and var_name not in matched_vars:
                phrase = doc[start:end].text.lower()
                code = self.reverse_mappings[var_name].get(phrase)
                if code:
                    extracted[var_name] = code
                    matched_vars.add(var_name)

        return extracted

    def fill_url(self, url_template: str, extracted_vars: Dict[str, str]) -> str:
        try:
            return self.root + url_template.format(**extracted_vars) # include root of url
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(f"Missing required variable in query: '{missing}'")