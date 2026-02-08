import requests
import json

base_url = "XXX"


url = f"{base_url}/api/chat"

payload = {
    "model": "gpt-oss:20b", 
    "messages": [
        {"role": "user", "content": "Welchen Informationsstand (Jahreszahl) hast du?"}
    ],
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=60)  
    response.raise_for_status()
    
    result = response.json()
    print("Antwort vom Modell:")
    print(result['message']['content'])
    
except Exception as e:
    print(f"Fehler beim API-Aufruf: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Status Code: {e.response.status_code}")
        print(f"Response: {e.response.text}")
