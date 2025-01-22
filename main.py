from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import io

app = FastAPI()

# Global variables for storing data and model
data = None
model = None
scaler = None

class PredictInput(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/upload")
async def upload(file: UploadFile):
    """Endpoint to upload a CSV file."""
    try:
        # Log file details
        print(f"Filename: {file.filename}")
        print(f"Content-Type: {file.content_type}")

        # Read the file content
        content = await file.read()

        # Check if the file is empty
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Decode and load into a DataFrame
        print("Reading CSV content...")
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        # Validate required columns
        required_columns = ["Temperature", "Run_Time", "Downtime_Flag"]
        print("Validating columns...")
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {', '.join(required_columns)}")

        # Add new feature
        df["Temperature_RunTime"] = df["Temperature"] * df["Run_Time"]

        # Log success
        print("File successfully processed!")
        global data
        data = df
        return {"message": "File uploaded and data processed successfully!"}
    except Exception as e:
        # Log the error and raise an HTTPException
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/train")
def train():
    """Endpoint to train the Logistic Regression model"""
    global data, model, scaler
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded for training.")
    try:
        # Prepare data
        X = data[["Temperature", "Run_Time", "Temperature_RunTime"]]
        y = data["Downtime_Flag"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression model
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Log confusion matrix for further analysis
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:", cm)

        # Save the model and scaler
        with open("model.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)

        return {"message": "Model trained successfully!", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
def predict(input_data: PredictInput, threshold: float = 0.6):
    """Endpoint to make predictions with configurable threshold"""
    global model, scaler
    if model is None or scaler is None:
        # Load model and scaler if available
        try:
            with open("model.pkl", "rb") as f:
                saved = pickle.load(f)
                model = saved["model"]
                scaler = saved["scaler"]
        except:
            raise HTTPException(status_code=400, detail="Model not trained. Train the model first.")

    try:
        # Make a prediction
        Temperature_RunTime = input_data.Temperature * input_data.Run_Time
        features = [[input_data.Temperature, input_data.Run_Time, Temperature_RunTime]]
        scaled_features = scaler.transform(features)
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = max(probabilities)
        prediction = "Yes" if confidence > threshold else "No"
        result = {"Downtime": prediction, "Confidence": confidence}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
