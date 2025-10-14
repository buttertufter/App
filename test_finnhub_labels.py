#!/usr/bin/env python3
"""Test script to see Finnhub labels."""

import sys
import os
sys.path.insert(0, '/workspaces/App')
sys.path.insert(0, '/workspaces/App/app')

from dotenv import load_dotenv
import requests

load_dotenv('/workspaces/App/.env')
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

base_url = "https://finnhub.io/api/v1"
headers = {"X-Finnhub-Token": FINNHUB_API_KEY}

income_url = f"{base_url}/stock/financials-reported?symbol=MSFT&freq=quarterly"
resp = requests.get(income_url, headers=headers)

data = resp.json().get('data', [])
if data:
    report = data[0]  # First report
    report_obj = report.get('report', {})
    
    print(f"Report date: {report.get('endDate')}")
    print("\n" + "="*60)
    print("INCOME STATEMENT (ic) LABELS:")
    print("="*60)
    ic = report_obj.get('ic', [])
    for item in ic:
        print(f"{item.get('label')}: {item.get('value')}")
    
    print("\n" + "="*60)
    print("BALANCE SHEET (bs) LABELS:")
    print("="*60)
    bs = report_obj.get('bs', [])
    for item in bs[:10]:  # First 10
        print(f"{item.get('label')}: {item.get('value')}")
    
    print("\n" + "="*60)
    print("CASH FLOW (cf) LABELS:")
    print("="*60)
    cf = report_obj.get('cf', [])
    for item in cf[:10]:  # First 10
        print(f"{item.get('label')}: {item.get('value')}")
