import requests

# Make a prediction
url = 'http://localhost:5000/predict'
files = {'file': open('data/SampleImage1.jpg', 'rb')}
response = requests.post(url, files=files)
predictions = response.json()

print(predictions)