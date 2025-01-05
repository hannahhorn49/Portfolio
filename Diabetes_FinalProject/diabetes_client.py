##############################################
# Programmer: Hannah Horn, Eva Ulrichsen
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #final project
# 12/9/24
# I did not attempt the bonus
# Description: This program contains API URL.
#########################

import requests
import json

# sample url with "unseen instance"
url = "http://127.0.0.1:5001/predict?age=22&a1c_level=5.5&glucose_level=110&hypertension=0.0&heart_disease=0.0"

response = requests.get(url = url)

# check the response's status code
print(response.status_code)
if response.status_code == 200:
    # STATUS OK
    # we can extract the prediction from the response's JSON text
    json_object = json.loads(response.text)
    print(json_object)
    pred = json_object["prediction"]
    print("prediction:", pred)