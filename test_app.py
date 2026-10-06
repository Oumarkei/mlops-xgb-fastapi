from urllib import response

from fastapi.testclient import  TestClient
from app import app

# TestClient crée un serveur virtuel instantané pour tester
client  = TestClient(app)

def test_sanity_check():
    response = client.get("/")
    assert response.status_code == 200 # Check for successful server response
    assert response.json()["status"] == "active API"
    assert response.json()["pipeline_loaded"] == True

def test_predict_valid_data():
    # On simule un payload JSON envoyé par un utilisateur
    payload = {
        "mean_radius": 14.13,
        "mean_texture": 19.3,
        "mean_perimeter": 92,
        "mean_area": 654.89,
        "mean_smoothness": 0.01,
        "mean_compactness": 0.01,
        "mean_concavity": 0.09
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "class_name" in data
    assert "confidence" in data
    assert data["prediction"] in [0,1]