import requests

# Make a prediction
url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
predictions = response.json()