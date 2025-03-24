# Trend-X-BTC API

Trend-X-BTC API provides endpoints for training and predicting any crypto price trends using deep learning.

## Endpoints

### Train Model
**Route:** `POST /train`  
**Request Body:**  
```json
{
  "symbol": "BTC"
}
```
**Description:** Trains the model for the specified cryptocurrency.

### Predict Prices
**Route:** `POST /predict`  
**Request Body:**  
```json
{
  "symbol": "BTC"
}
```
**Description:** Predicts the next 30 days' price trends for the specified cryptocurrency.

## Usage
Send a `POST` request with the coin symbol to the respective endpoint.
