import requests

url = "http://localhost:9696/predict"
input_data = {"text": "I really like this!"}

response = requests.post(url, json=input_data)
print(response.json())
