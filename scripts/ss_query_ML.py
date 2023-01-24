import requests
import json

key = 'X-API-Key'
secret = 'fGcuQ7hOHg1SSujLGohaS5Ei9lNjmxQx9112AWiI'
url = 'https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&fields=title,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,fieldsOfStudy,authors,url,abstract&limit=100&offset=6000&next=100'

headers = {
    'Accept': '*/*',
    key: secret
}

response = requests.get(url, headers=headers)

with open('600_samples_data.json', 'w') as f:
    json.dump(response.json(), f)
