"""Google Sheets access layer for the WhatsApp volunteer bot.

Expects a spreadsheet with three tabs:

  Volunteers    | Name | Phone | Active |
  Opportunities | OpportunityID | Task | Details | ConfirmBy | EventDate |
  Responses     | OpportunityID | Name | Phone | Status | MessageSentAt | RepliedAt | RawReply |
"""

import os

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def get_spreadsheet():
    creds_file = os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"]
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    credentials = Credentials.from_service_account_file(creds_file, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client.open_by_key(sheet_id)


def get_volunteers(spreadsheet):
    """Return active volunteers as a list of {name, phone} dicts."""
    rows = spreadsheet.worksheet("Volunteers").get_all_records()
    volunteers = []
    for row in rows:
        active = str(row.get("Active", "yes")).strip().lower()
        if active in ("", "yes", "y", "true", "1"):
            volunteers.append({"name": str(row["Name"]).strip(), "phone": str(row["Phone"]).strip()})
    return volunteers


def get_opportunity(spreadsheet, opportunity_id):
    """Look up a single opportunity by OpportunityID."""
    for row in spreadsheet.worksheet("Opportunities").get_all_records():
        if str(row.get("OpportunityID", "")).strip() == str(opportunity_id):
            return row
    raise ValueError(f"No opportunity found with ID '{opportunity_id}'")


def log_message_sent(spreadsheet, opportunity_id, name, phone, sent_at):
    worksheet = spreadsheet.worksheet("Responses")
    worksheet.append_row(
        [opportunity_id, name, phone, "Pending", sent_at, "", ""],
        value_input_option="USER_ENTERED",
    )


def record_reply(spreadsheet, phone, status, replied_at, raw_reply):
    """Update the most recent 'Pending' Responses row for this phone number.

    Returns True if a row was updated, False if no pending row matched.
    """
    worksheet = spreadsheet.worksheet("Responses")
    header = worksheet.row_values(1)
    records = worksheet.get_all_records()

    status_col = header.index("Status") + 1
    replied_col = header.index("RepliedAt") + 1
    raw_col = header.index("RawReply") + 1

    target_row = None
    for idx, row in enumerate(records, start=2):  # row 1 is the header
        if str(row.get("Phone", "")).strip() == phone and row.get("Status") == "Pending":
            target_row = idx  # keep scanning so the *latest* pending row wins

    if target_row is None:
        return False

    worksheet.update_cell(target_row, status_col, status)
    worksheet.update_cell(target_row, replied_col, replied_at)
    worksheet.update_cell(target_row, raw_col, raw_reply)
    return True
