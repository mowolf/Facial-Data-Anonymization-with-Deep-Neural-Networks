import json
import time
import urllib.request
import requests
from tqdm import tqdm
from telegram_bot import send_alert


def downloadIds(continue_from_id=68520):
    print("Reading .json")
    with open('/home/mo/experiments/masterthesis/flickr/ffhq-dataset-v2.json', 'r') as f:
        data = json.load(f)
    print("Processing data")

    for key, item in tqdm(data.items()):
        if int(key) > continue_from_id:
            # if int(key) % 1000 == 0:
            #     print("Waiting 10 minutes to not exeed api limits.")
            #     time.sleep(600)
            time.sleep(1)
            url = item["metadata"]["photo_url"][:-1]
            id = {"flickr_id": url[url.rfind("/") + 1:], "ffhq_id": key}
            downloadFile(id)


def downloadFile(id):
    url = f'https://www.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key=c6a2c45591d4973ff525042472446ca2&photo_id={id["flickr_id"]}&format=rest'
    r = requests.get(url)
    # For successful API call, response code will be 200 (OK)
    if (r.ok):
        content = str(r.content)
        end = content.find("_o.jpg")
        start = content[0:end].rfind("source=")
        photo_url = content[start + 8:end + 6].replace('https', 'http')
        # Download image from the url and save
        try:
            with urllib.request.urlopen(photo_url) as d, open(f'/mnt/raid5/mo/ffhq_redownloaded/{id["ffhq_id"]}.png',
                                                              "wb") as opfile:
                data = d.read()
                opfile.write(data)
        except ValueError:
            pass
        except urllib.error.HTTPError:
            alert = f'504 Timed Out for ID: {id["ffhq_id"]}'
            send_alert(alert)
            print(alert)
            time.sleep(1)
        except Exception as e:
            send_alert(e)


if __name__ == '__main__':
    ids = downloadIds()
