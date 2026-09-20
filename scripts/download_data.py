"""Download original UCI weekly quantities with no account required."""
from pathlib import Path
import urllib.request, io, zipfile, hashlib
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/396/sales+transactions+dataset+weekly.zip'
if __name__ == '__main__':
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        content = archive.read('Sales_Transactions_Dataset_Weekly.csv')
    path = ROOT / 'data/raw/Sales_Transactions_Dataset_Weekly.csv'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    print('SHA256:', hashlib.sha256(content).hexdigest())
