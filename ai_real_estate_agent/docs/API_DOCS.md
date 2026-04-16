# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider adding API key authentication.

## Endpoints

### Health Check

Check if the API is running and healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Predict Property Price

Predict the price of a property based on its features.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "property_features": {
    "lot_area": 8000,
    "year_built": 2000,
    "year_remod_add": 2010,
    "mas_vnr_area": 100,
    "bsmt_unf_sf": 500,
    "total_bsmt_sf": 1000,
    "first_flr_sf": 1200,
    "garage_area": 500,
    "living_area": 2000
  },
  "include_explanation": true
}
```

**Response (200):**
```json
{
  "predicted_price": 185000.50,
  "confidence_interval": null,
  "explanation": "Based on the property features provided, this mid-range residential property in a standard market shows a predicted value of $185,000. The valuation factors in the lot size, construction age, and finished living space. Comparable properties in this segment typically range from $170,000 to $200,000."
}
```

**Error Responses:**
- `400` - Invalid request or prediction failed
- `503` - Model not loaded
- `500` - Server error

---

### Query LLM

Ask the AI assistant a question about real estate.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "question": "What factors affect residential property values?",
  "context": null
}
```

**Response (200):**
```json
{
  "answer": "Residential property values are influenced by several key factors: location (proximity to schools, amenities), property size and condition, market conditions, interest rates, local economic factors, neighborhood demand, and recent comparable sales. Supply and demand dynamics also play a crucial role.",
  "source": "openai"
}
```

**Error Responses:**
- `500` - Server error

---

## Interactive API Documentation

Automatically generated documentation available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Rate Limiting

Currently not implemented. Consider adding in production.

## Pagination

Not applicable for current endpoints.

## Error Handling

All errors return JSON with the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## CORS

CORS is enabled for all origins (`*`) in development. Configure appropriately for production.
