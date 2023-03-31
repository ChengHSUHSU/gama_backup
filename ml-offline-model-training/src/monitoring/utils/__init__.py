import requests
import json


def get_model_description(url):
    content = requests.get(url, stream=True).content
    content = json.loads(content).get('hits', [])
    return content
