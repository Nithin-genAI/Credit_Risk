from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Drifting Oracle"


def test_predict_includes_shap_factors():

    response = client.post(
        "/predict",
        json={
            "application_id": "APP-10001",
            "name": "Rahul",
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -12000,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert "top_factors" in result
    assert len(result["top_factors"]) == 5
    assert all(
        set(factor) == {"feature", "impact", "shap_value"}
        for factor in result["top_factors"]
    )