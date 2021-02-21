import requests

resp=requests.post("http://localhost:5000/chat",params={'sentence':'Can you create a excel file?'})

print (resp.text)