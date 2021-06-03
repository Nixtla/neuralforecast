import os
import sys

import requests
from bs4 import BeautifulSoup


def extract_report(html):
    return BeautifulSoup(html, 'html.parser').select('code')[0]

token = os.getenv('GITHUB_TOKEN')
pr_number = os.getenv('PR_NUMBER')

with open(sys.argv[1], 'rt') as f:
    html_master = f.read()
with open(sys.argv[2], 'rt') as f:
    html_pr = f.read()

master_report = extract_report(html_master)
pr_report = extract_report(html_pr)
url = f'https://api.github.com/repos/Nixtla/nixtlats/issues/{pr_number}/comments'
headers = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {token}',
}
message = f'master:\n{master_report}\npr:\n{pr_report}'
data = {'body': message}
result = requests.post(url=url, headers=headers, json=data)
status_code = result.status_code
if status_code != 201:
    raise Exception(f'Received status code {status_code}')
