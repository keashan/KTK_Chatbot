import requests

resp=requests.post("http://http://143.110.217.198:80/chat",params={'sentence':'Can you create a excel file?'})

print (resp.text)