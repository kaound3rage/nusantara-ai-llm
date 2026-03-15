import requests

url = "https://5l8krs6cdd.execute-api.us-east-1.amazonaws.com/prod/recommend"

data = {
    "user_id": "USR00001",
    "top_n": 6
}

response = requests.post(url, json=data, timeout=30)

print(response.json())