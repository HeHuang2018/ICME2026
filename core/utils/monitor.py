
import pandas as pd
import matplotlib.pyplot as plt
import os,re,time,datetime,base64,requests,json,json,hashlib

from urllib import parse
from loguru import logger as log
from requests_toolbelt import MultipartEncoder

WEBHOOKURL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=b52fbce4-0a1e-4935-8bea-1565bf10a5a3"
def send_temp(cfg,new_results):
    headers = {"content-type": "application/json"}
    data={
        "msgtype":"template_card",
        "template_card":{
            "card_type":"text_notice",
            "source":{
                "desc":"TTA Results",
                "desc_color":0
            },
            "main_title":{
                "title":"TTA results",
                "desc":f"Start from {cfg.bash_file_name}"
            },
            "sub_title_text":"Detail",
            "horizontal_content_list":[
                {
                    "keyname":"TTA Method",
                    "value": f"{cfg.ADAPTER.NAME}",
                },
                {
                    "keyname":"Dataset",
                    "value": f"{cfg.CORRUPTION.DATASET}",
                },
                {
                    "keyname":"Seed",
                    "value": f"{cfg.SEED}",
                },
                {
                    "keyname":"Batch Size",
                    "value": f"{cfg.TEST.BATCH_SIZE}",
                },
                {
                    "keyname":"Severity",
                    "value": f"{cfg.CORRUPTION.SEVERITY}",
                },
                {
                    "keyname":"Order",
                    "value": f"{cfg.CORRUPTION.ORDER_NUM}",
                }
            ],
            "quote_area":{
                "title":"Note",
                "quote_text": f"{cfg.NOTE}"
            },
            "card_action":{
                "type":1,
                "url":"/",
                "appid":"APPID",
                "pagepath":"PAGEPATH"
            }
        }
    }
    try:
        result = requests.post(WEBHOOKURL, headers=headers, json=data)
    except Exception as e:
        log.info("Requset Failed:", e)

def send_msg(msg):
    headers = {"content-type": "application/json"}
    msg = {"msgtype": "text","text": {"content": msg}}
    try:
        result = requests.post(WEBHOOKURL, headers=headers, json=msg)
        return True
    except Exception as e:
        # print("Requset Failed:", e)
        return False

def format_df(df):
    df = df[['method', 'dataset', 'batch_size', 'seed', 'model', 'severity', 'Avg','note']]
    # df = df[df['dataset'] == 'cifar100']
    df = df.sort_values(by='Avg', ascending=False)
    if len(df) > 40:
        df = df.iloc[:40, :]
    
    df = df.round(2)
    df = df.reset_index(drop=True)
    df.to_csv("./old/temp.csv", index=False)
    return df

def df_to_img(all_results_df,imgPath,new_results):
    try:
        df = format_df(all_results_df)
        highlight_index = df[
            (df['note'] == new_results['note']) &
            (df['method'] == new_results['method']) &
            (df['dataset'] == new_results['dataset']) &
            (df['seed'] == new_results['seed']) &
            (df['batch_size'] ==  new_results['batch_size']) &
            (df['severity'] == new_results['severity'])
        ].index.tolist()[0]


        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colWidths=[0.2]*len(df.columns))
        for i in range(len(df.columns)):
            table[(highlight_index+1, i)].set_facecolor("#ffcccc")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        plt.savefig(imgPath)
    except:
        pass

def send_img(imgPath):
    with open(imgPath,"rb") as f:
        fd=f.read()
        base64Content=str(base64.b64encode(fd),"utf-8")
    with open(imgPath,"rb") as f:
        fd=f.read()
        md = hashlib.md5()
        md.update(fd)
        md5Content = md.hexdigest()
    headers = {"content-type": "application/json"}
    msg = {"msgtype": "image","image": {"base64": base64Content,"md5": md5Content}}
    try:
        result = requests.post(WEBHOOKURL, headers=headers, json=msg)
        return True
    except Exception as e:
        send_msg(f"send_img faile {str(e)}")
        return False

def upload_file(filepath):
    params = parse.parse_qs( parse.urlparse( WEBHOOKURL ).query )
    webHookKey=params['key'][0]
    upload_url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={webHookKey}&type=file'
    headers = {"Accept": "application/json, text/plain, */*", "Accept-Encoding": "gzip, deflate",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36"}
    filename = os.path.basename(filepath)
    try:
        multipart = MultipartEncoder(
            fields={'filename': filename, 'filelength': '', 'name': 'media', 'media': (filename, open(filepath, 'rb'), 'application/octet-stream')},
            boundary='-------------------------acebdf13572468'
        )
        headers['Content-Type'] = multipart.content_type
        resp = requests.post(upload_url, headers=headers, data=multipart)
        json_res = resp.json()
        if json_res.get('media_id'):
            return json_res.get('media_id')
    except Exception as e:
        send_msg(f"error {str(e)}")
        return ""

def send_file(filepath):
    media_id = upload_file(filepath)
    msg={"msgtype":"file","file":{"media_id":media_id}}
    try:
        result = requests.post(WEBHOOKURL, headers = {
            "content-type": "application/json"
        }, json=msg)
        return True
    except Exception as e:
        send_msg(f"error {str(e)}")
        return False
