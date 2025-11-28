import requests
import json

url = "http://127.0.0.1:5000/upload"

payload = {
    "gateway_id": "A1",
    "value": 123,
}

response = requests.post(url, json=payload)

print("Server response:", response.text)