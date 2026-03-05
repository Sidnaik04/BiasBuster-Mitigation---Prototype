import requests

API_URL = "http://127.0.0.1:8000"


def _response_json(response: requests.Response):
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text or "Invalid JSON response"}

    if isinstance(payload, dict):
        payload.setdefault("http_status", response.status_code)

    return payload


def upload_files(dataset, model):
    files = {
        "dataset": dataset,
        "model": model
    }

    r = requests.post(f"{API_URL}/upload/", files=files)
    return _response_json(r)


def run_baseline(upload_id, target, sensitive):

    params = {
        "upload_id": upload_id,
        "target_column": target,
        "sensitive_attribute": sensitive
    }

    r = requests.post(f"{API_URL}/baseline/", params=params)
    return _response_json(r)


def recommend_strategy(upload_id, baseline_metrics):
    r = requests.post(
        f"{API_URL}/mitigation/recommend",
        params={"upload_id": upload_id},
        json=baseline_metrics,
    )

    if r.status_code == 422:
        payload = {
            "upload_id": upload_id,
            "baseline_metrics": baseline_metrics,
        }
        r = requests.post(
            f"{API_URL}/mitigation/recommend",
            json=payload,
        )

    return _response_json(r)


def apply_mitigation(payload):
    params = {
        "upload_id": payload["upload_id"],
        "target_column": payload["target_column"],
        "sensitive_attribute": payload["sensitive_attribute"],
        "strategy": payload["strategy"],
    }
    strategy_config = payload.get("strategy_config", {})

    r = requests.post(
        f"{API_URL}/mitigation/apply",
        params=params,
        json=strategy_config,
    )

    if r.status_code == 422:
        r = requests.post(
            f"{API_URL}/mitigation/apply",
            json=payload,
        )

    return _response_json(r)

def auto_rank_strategies(results):

    r = requests.post(
        f"{API_URL}/auto-mitigation/rank",
        json=results
    )

    return _response_json(r)