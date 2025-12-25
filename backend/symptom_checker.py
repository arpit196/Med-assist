import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Required for development to allow React to talk to Flask
#from google import genai
#from google.genai import types
import spacy
import scispacy
import requests
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Optional, Callable
# --- Configuration ---
# NOTE: Replace "YOUR_GEMINI_API_KEY" with your actual key, or better, set it as an environment variable.
# For local testing, you can uncomment and set the key directly:
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# If you don't set it below, the script will look for an environment variable named GEMINI_API_KEY.

# Initialize the Flask App
app = Flask(
    __name__,
    static_folder='build', # The directory containing your React static files
    static_url_path=''
)

#react_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))

# Enable CORS for all origins during development
CORS(app) 


from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Flatten, BatchNormalization
import keras

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import pathlib
from dotenv import load_dotenv, find_dotenv
from fuzzywuzzy import process

#is_loaded = load_dotenv(find_dotenv(), override=True)
#print("Env loaded:", is_loaded)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full relative path from the script location
PRECAUTION_FILE = os.path.join(BASE_DIR, 'disease-symptom', 'Disease precaution.csv')
SYMPTOM_FILE = os.path.join(BASE_DIR, 'disease-symptom', 'DiseaseAndSymptoms.csv')

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent'
#print('API_KEY')
#print(GEMINI_API_KEY)
from google import genai

ACTIVE_CHATS = {} 

def get_or_create_chat(user_id):
    """Retrieves an existing chat session or creates a new one."""
    global ACTIVE_CHATS
    
    if user_id not in ACTIVE_CHATS:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            raise ValueError(f"Error initializing client: {e}")

        # System Instruction for the AI's persona and context
        system_instruction = (
            "You are a compassionate, adaptive, and highly knowledgeable medical assistant. "
            "Maintain context from previous turns. Base all recommendations on the user's "
            "provided profile (age, sex, weight) and previous symptoms."
        )

        chat = client.chats.create(
            model='gemini-2.5-flash', 
            system_instruction=system_instruction
        )
        ACTIVE_CHATS[user_id] = chat
        return chat
    else:
        return ACTIVE_CHATS[user_id]

class l12Regularizer(keras.regularizers.Regularizer):
    """
    Custom regularizer that applies a penalty based on the L1 and L2 norms of the weights.
    Inherits from `keras.regularizers.Regularizer`.
    """

    def __init__(self, l1=0.06):
        # Store the coefficients for serialization
        self.l1 = l1

    def __call__(self, weights):
        """
        Calculates the regularization penalty for the given weights.
        This method is called by the Keras model during training.

        Args:
            weights (tf.Tensor): The weights of the layer to regularize.

        Returns:
            tf.Tensor: The calculated regularization penalty.
        """
        # Calculate the L1 penalty ### *tf.stop_gradient(tf.convert_to_tensor(1.0/(1.0+0.01*correlation**2),dtype=tf.float32)
        l1_penalty = self.l1 * tf.reduce_sum(tf.sqrt(tf.abs(weights)+1e-6))

        return l1_penalty

    def get_config(self):
        """
        Returns the configuration of the regularizer.
        This is necessary to serialize the regularizer when saving the model.

        Returns:
            dict: A dictionary containing the regularizer's configuration.
        """
        return {'l1': float(self.l1)}

def robustLasso():
    inputs = Input(shape=(131,))
    x = keras.layers.Dense(130,activation='relu',kernel_regularizer=l12Regularizer(0.005))(inputs)
    x = keras.layers.Dense(41,activation='softmax',kernel_regularizer=l12Regularizer(0.005))(x) #l12Regularizer(0.04)
    return keras.Model(inputs=inputs,outputs=x)

# Initialize the Gemini Client
'''try:
    # Client will automatically look for the GEMINI_API_KEY environment variable
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    # Fallback for local testing if you set the key directly above
    # client = genai.Client(api_key=GEMINI_API_KEY)'''

def load_scispacy_model(model_name="en_ner_bc5cdr_md"):
    """
    Loads the specified scispaCy model.
    """
    try:
        # Load the model using spacy.load
        nlp = spacy.load(model_name)
        print(f"✅ Successfully loaded model: {model_name}")
        return nlp
    except OSError:
        # Handle the case where the model is not installed or downloaded
        print("❌ Error: scispaCy model not found.")
        print(f"Please ensure you run the installation command: pip install {model_name} (see file comments)")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def extract_clinical_symptoms(text, nlp):
    """
    Processes text using the scispaCy model and extracts entities
    labeled as 'DISEASE', which are often symptoms or clinical conditions.
    """
    # Process the raw text
    
    print('Hi2')
    doc = nlp(text)
    print(doc)
    # Filter entities based on the 'DISEASE' label
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    print(symptoms)
    return symptoms

def map_to_closest_symptom(symptom : str, known_symptom_list: List[str], threshold=70):
    """
    Maps a given symptom string to the closest known symptom using fuzzy matching.
    Returns None if similarity is below the threshold.
    Args: 
        symptom : a single symptom as a string
        known_symptom_list: a list of possible symptoms that are utilized as features by the ML model.
    Returns:
        The closest matching symptom from the known_symptom_list that resembles the symptom the most.
        Fuzzy matching is used here to match the strings.
    """
    match, score = process.extractOne(symptom, known_symptom_list)
    if score >= threshold:
        return match
    else:
        return None
    
def keyword_to_symptom_matcher(keywords):
    features = np.zeros(131)
    df = pd.read_csv(SYMPTOM_FILE)
    data2=df.drop_duplicates().reset_index(drop=True)
    print(df.head)
    all_symptoms=set()
    for i in data2.columns:
        if i=='Disease' or pd.isna(i) or not isinstance(i,str):
            continue
        all_symptoms=all_symptoms.union(set(data2[i].astype(str).str.strip().unique()))
    all_symptoms = {x for x in all_symptoms if not isinstance(x, float) or not math.isnan(x)}

    all_symptoms_ordered = sorted(list(all_symptoms))

    dict={}
    for i in range(len(all_symptoms_ordered)):
        dict[all_symptoms_ordered[i]]=i

    for keyword in keywords:
        for i, symptom in enumerate(list(dict.keys())):
            if not isinstance(symptom,str):
                continue
            if keyword.lower() == symptom.replace('_',' ').lower():
                #print('Hi')
                features[dict[symptom]] = 1
    return features

def predict_disease_and_precautions(keywords):
    care = pd.read_csv(PRECAUTION_FILE)
    df = pd.read_csv(SYMPTOM_FILE)

    feats = keyword_to_symptom_matcher(keywords)
    
    print(feats)
    aLasso = robustLasso()
    aLasso.load_weights('symptom_evaluator2')
    print('X')
    predicted_disease = aLasso.predict(feats[np.newaxis,:])
    ohe = LabelEncoder()
    P = df[["Disease"]]
    P_enc = ohe.fit_transform(P)
    disease=str(ohe.inverse_transform([np.argmax(predicted_disease)])[0])
    print(disease)
    print('Predicted disease is :'+disease)
    print('Precautions to be taken are :'+", ".join(care[care['Disease']==disease].values[0][1:]))
    return disease, care[care['Disease']==disease].values[0][1:]


@app.route('/api/diagnosis', methods=['POST'])
def get_diagnosis():
    """
    Handles the symptom submission, calls Gemini API, and returns the analysis.
    """
    #if not client:
    #    return jsonify({"error": "AI Client not initialized."}), 500

    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        age = data.get('age', 0)
        sex = data.get('sex', 'unspecified')
        height = data.get('height',100)
        weight = data.get('weight',100)
        message = data.get('message','')
        print(symptoms)
        if not symptoms or len(symptoms) < 20:
            return jsonify({"error": "Symptom description is too short or missing."}), 400
        nlp = load_scispacy_model()
        print('Hi2')
        extracted_symptoms = extract_clinical_symptoms(symptoms, nlp)
        predicted_disease,care = predict_disease_and_precautions(extracted_symptoms)
        print('Hi3')
        text = 'Given the predicted disease for a patient:' + str(predicted_disease) + 'with symptoms' + str(extracted_symptoms) + 'and recommendations' + str(care) + 'suggest lifestyle and diet habits that can help him manage the disease.'
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Error initializing client. Make sure your API key is set as an environment variable: {e}")
        model_name = 'gemini-2.5-flash'
        response = client.models.generate_content(
            model=model_name,
            contents=text
        )
        print(response.text)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON input: {e}"}), 400
    

    # --- Gemini Prompt Construction ---
    
    return jsonify({
            "text": str(predicted_disease),
            "gemini": response.text
        })

@app.route('/privacy')
def privacy():
    return send_from_directory('policies', 'privacy.html')

@app.route('/terms')
def terms():
    return send_from_directory('policies', 'terms.html')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):#os.path.exists(os.path.join(react_build_dir, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # If path not found, return React's index.html (for client-side routing)
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Flask runs on http://127.0.0.1:5000 by default
    app.run(debug=True, port=5000)
