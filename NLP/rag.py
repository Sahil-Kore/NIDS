import os
import torch
import joblib
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, List

# --- Pydantic Models for API ---

class FeatureInput(BaseModel):
    """Defines the input for raw features."""
    features: Dict[str, float]

class StringInput(BaseModel):
    """Defines the input for a pre-formatted feature string."""
    question: str

class ExplanationResponse(BaseModel):
    """Defines the successful response from the API."""
    prediction: str
    explanation: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cybersecurity RAG Explainer",
    description="API to get explanations for network traffic classifications",
    version="1.0.0"
)

# --- Global Variables to hold models and data ---
le = None
finetuned_tokenizer = None
finetuned_model = None
llm = None
features = []
means = None
std = None
explanation_chain = None

# --- Helper Functions (from your rag.py) ---

def get_distilbert_prediction(text_input: str) -> str:
    """Runs prediction using the loaded DistilBERT model."""
    inputs = finetuned_tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        logits = finetuned_model(**inputs).logits
    
    prediction_id = torch.argmax(logits, dim=-1).item()
    prediction_name = le.inverse_transform([prediction_id])[0]
    return prediction_name

def create_feature_string(row: Dict[str, float], features_list: List[str], means_series: pd.Series, std_series: pd.Series) -> str:
    """Converts a dictionary of features into a descriptive string."""
    string = "Label with:\n"
    for attr in features_list:
        value = row.get(attr)
        
        if value is None:
            string += f" {attr}: not provided\n"
            continue

        std_val = std_series.get(attr)
        mean_val = means_series.get(attr)

        if pd.isna(std_val) or std_val == 0 or pd.isna(mean_val):
            string += f" {attr}: balanced (no variance or mean)\n"
            continue

        if value > mean_val + 1.5 * std_val: string += f" {attr}: extremely high \n"
        elif value > mean_val + 1.0 * std_val: string += f" {attr}: high \n"
        elif value > mean_val + 0.75 * std_val: string += f" {attr}: slightly high\n"
        elif value < mean_val - 1.5 * std_val: string += f" {attr}: extremely low\n"
        elif value < mean_val - 1.0 * std_val: string += f" {attr}: low\n"
        elif value < mean_val - 0.75 * std_val: string += f" {attr}: slightly low\n"
        else: string += f" {attr}: balanced\n"
    return string

def run_prediction(input_dict: Dict[str, Any]) -> str:
    """Wrapper for chain to run prediction."""
    q = input_dict["question"]
    return get_distilbert_prediction(q)

# --- Startup Event to Load Models ---

@app.on_event("startup")
async def load_models_and_data():
    """
    Loads all models, data, and builds the LangChain on app startup.
    """
    global le, finetuned_tokenizer, finetuned_model, llm
    global features, means, std, explanation_chain

    print("--- Loading models and data at startup... ---")
    
    try:
        # 1. Load Label Encoder
        le_path = './label_encoder.joblib'
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"LabelEncoder not found at {le_path}")
        le = joblib.load(le_path)
        print("LabelEncoder loaded.")

        # 2. Load DistilBERT model and tokenizer
        model_path = "./distilbert_multiclass_model/checkpoint-6000"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Finetuned model not found at {model_path}")
        finetuned_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        finetuned_model.eval()
        print("DistilBERT model and tokenizer loaded.")

        # 3. Initialize LLM
        llm = ChatOllama(model="phi3:mini", temperature=0)
        print("ChatOllama (phi3:mini) initialized.")

        # 4. Load data for feature string generation
        test_data_path = "../Data/filtered_data/test.parquet"
        train_data_path = "../Data/filtered_data/train.parquet"
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Train data not found at {train_data_path}")

        test_df = pd.read_parquet(test_data_path)
        features = [col for col in test_df if col != "Label"]
        
        train_df = pd.read_parquet(train_data_path)
        
        benign_label_int = le.transform(["Benign"])[0]
        benign_df = train_df[train_df["Label"] == benign_label_int]
        means = benign_df[features].mean()
        std = benign_df[features].std()
        print("Feature stats (means, std) calculated.")

        # 5. Define LangChain
        template = """
You are a senior cybersecurity analyst. Your job is to explain *why* a classification model's prediction makes sense, given the input features.

Here is the data you have:

---
[INPUT FEATURES]
The model observed the following network behavior:
{question}
---
[MODEL PREDICTION]
The model classified this behavior as:
{prediction}
---

[YOUR TASK]
Please write a brief, one-paragraph explanation for why the model's prediction is plausible.
Analyze the [INPUT FEATURES] and explain how they might be characteristic of the [MODEL PREDICTION].
Start your answer with "The model's prediction of '{prediction}' is plausible because..."
"""
        prompt = ChatPromptTemplate.from_template(template)

        # Build chain to return both prediction and explanation
        explanation_chain = (
            RunnablePassthrough.assign(
                prediction=RunnableLambda(run_prediction)
            ).assign(
                explanation=prompt | llm | StrOutputParser()
            )
        )
        print("--- All models and data loaded successfully. API is ready. ---")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("API will not be functional. Please check file paths.")
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Health check to see if models loaded."""
    if explanation_chain and le and features:
        return {"status": "ok", "message": "Models and data are loaded."}
    else:
        return {"status": "error", "message": "Models or data failed to load. Check server logs."}

@app.post("/explain_from_features", response_model=ExplanationResponse)
async def explain_from_features(input: FeatureInput):
    """
    Generates a classification and explanation from raw numerical features.
    """
    if not explanation_chain:
        raise HTTPException(status_code=503, detail="Server is not ready. Models not loaded.")
    
    try:
        # 1. Create the feature string from the input dictionary
        input_string = create_feature_string(input.features, features, means, std)
        
        # 2. Run the explanation chain
        result = await explanation_chain.ainvoke({"question": input_string})
        
        # 3. Return the structured response
        return ExplanationResponse(
            prediction=result["prediction"],
            explanation=result["explanation"]
        )
    except Exception as e:
        print(f"Error during feature explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/explain_from_string", response_model=ExplanationResponse)
async def explain_from_string(request: StringInput):
    """
    Generates a classification and explanation from a pre-formatted feature string.
    """
    if not explanation_chain:
        raise HTTPException(status_code=503, detail="Server is not ready. Models not loaded.")

    try:
        # Run the chain directly with the provided string
        result = await explanation_chain.ainvoke({"question": request.question})
        
        return ExplanationResponse(
            prediction=result["prediction"],
            explanation=result["explanation"]
        )
    except Exception as e:
        print(f"Error during string explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Main execution ---
if __name__ == "__main__":
    """
    Run the API server.
    
    To run:
    1. Make sure all models/data files are in the correct relative paths.
    2. Make sure you have an Ollama server running with the 'phi3:mini' model.
    3. Run: uvicorn main:app --reload --port 8000
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)