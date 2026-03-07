import gspread
from googleapiclient.discovery import build

SPREADSHEET_ID = 'Replace Spread Sheet id'
RANGE_NAME = 'Sheet1!A:F'
API_KEY = 'Get the api key'

def authenticate_sheets(api_key):
    return build('sheets', 'v4', developerKey=api_key).spreadsheets()

if __name__ == "__main__":
    sheets = authenticate_sheets(API_KEY)
    result = sheets.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])
    for row in values:
        print(f'{row[0]} | {row[1]} | {row[2]}')


