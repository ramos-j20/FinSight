import urllib.request
import urllib.error

try:
    print("Testing /eval/routing-stats internally...")
    req = urllib.request.Request('http://backend:8000/eval/routing-stats')
    res = urllib.request.urlopen(req, timeout=5)
    print("STATUS:", res.status)
    print("BODY:", res.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code)
    print("BODY:", e.read().decode('utf-8'))
except Exception as e:
    print("OTHER ERROR:", e)
