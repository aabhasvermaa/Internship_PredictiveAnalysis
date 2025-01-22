# Predictive API with FastAPI

This project provides a FastAPI-based API for training a Logistic Regression model to predict machine downtime based on temperature and runtime inputs. The API supports uploading datasets, training the model, and making predictions.

---

## Features
- Upload a dataset via `/upload`.
- Train a Logistic Regression model using `/train`.
- Make predictions with `/predict`.
- Automatically handles feature scaling and adds derived features.

---

## Requirements

1. Python 3.7+
2. Required Python packages:
   - `fastapi`
   - `uvicorn`
   - `pandas`
   - `scikit-learn`
   - `pickle`

Install the required packages using:
```bash
pip install fastapi uvicorn pandas scikit-learn
```

---

## Setup Instructions

### 1. Clone the Repository
If you have the project as a zip file or repository, extract or clone it into a folder.

### 2. Run the API
Navigate to the project folder in the terminal and start the FastAPI server:
```bash
python predictive_api_assignment.py
```
By default, the API will run on `http://0.0.0.0:8000`.

### 3. Use Swagger UI for Testing
1. Open your browser and navigate to the **Swagger UI** at:
   ```
   http://127.0.0.1:8000/docs
   ```
2. You will see interactive documentation for all the API endpoints.

---

## API Endpoints

### 1. Upload Dataset
**Endpoint:** `/upload`
- **Method:** `POST`
- **Description:** Upload a CSV file to prepare for training.
- **Required Columns:**
  - `Temperature`
  - `Run_Time`
  - `Downtime_Flag`
- **Steps to Use in Swagger UI:**
  1. Click on the `/upload` endpoint.
  2. Select the "Try it out" button.
  3. Upload a CSV file using the file picker.
  4. Click "Execute" to upload the file.

### 2. Train Model
**Endpoint:** `/train`
- **Method:** `POST`
- **Description:** Train the Logistic Regression model using the uploaded dataset.
- **Steps to Use in Swagger UI:**
  1. Click on the `/train` endpoint.
  2. Select the "Try it out" button.
  3. Click "Execute" to train the model.
- **Response Example:**
  ```json
  {
      "message": "Model trained successfully!",
      "metrics": {
          "accuracy": 0.85,
          "f1_score": 0.80
      }
  }
  ```

### 3. Make Predictions
**Endpoint:** `/predict`
- **Method:** `POST`
- **Description:** Predict downtime based on `Temperature` and `Run_Time`.
- **Steps to Use in Swagger UI:**
  1. Click on the `/predict` endpoint.
  2. Select the "Try it out" button.
  3. Enter the input values for `Temperature` and `Run_Time` in JSON format:
     ```json
     {
         "Temperature": 85,
         "Run_Time": 120
     }
     ```
  4. Click "Execute" to get the prediction.
- **Response Example:**
  ```json
  {
      "Downtime": "Yes",
      "Confidence": 0.85
  }
  ```
---

## Testing the API with Swagger UI

### Steps:
1. Navigate to `http://127.0.0.1:8000/docs` in your browser.
2. Use the `/upload` endpoint to upload your dataset.
3. Use the `/train` endpoint to train the model.
4. Use the `/predict` endpoint to test predictions.

Each endpoint in Swagger UI provides an interactive "Try it out" button to make testing straightforward.

---

## Troubleshooting

1. **Missing Columns in Dataset:** Ensure the CSV has all required columns.
2. **Model Not Trained:** Train the model before making predictions.
3. **API Not Starting:** Verify Python installation and required packages.
