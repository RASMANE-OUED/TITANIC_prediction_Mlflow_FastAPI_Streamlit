from fastapi.testclient import TestClient
from api import app  # Remplacez par le nom correct de votre fichier

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "pclass": 1,
        "sex": "female",
        "age": 26,
        "sibsp": 0,
        "parch": 0,
        "fare": 80.0,
        "embarked": "S"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability_survival" in data
    assert data["prediction"] in [0, 1]
