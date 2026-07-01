#!/usr/bin/env python3
"""Send a WhatsApp volunteering opportunity to every active volunteer.

Usage:
    python send_opportunity.py <OpportunityID>

Reads volunteers and the opportunity from the Google Sheet, sends each
volunteer a personalized WhatsApp message via Twilio, and logs a
'Pending' row per volunteer in the Responses tab so replies can be
matched up later by webhook_server.py.
"""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from twilio.rest import Client

from sheets_client import get_opportunity, get_spreadsheet, get_volunteers, log_message_sent

load_dotenv()


def build_message(volunteer_name, opportunity):
    details = opportunity.get("Details", "")
    return (
        f"Hi {volunteer_name}! \U0001F44B\n\n"
        f"We have a new volunteering opportunity for you:\n"
        f"*{opportunity['Task']}*\n"
        f"{details}\n\n"
        f"\U0001F4C5 Event date: {opportunity['EventDate']}\n"
        f"⏳ Please confirm by: {opportunity['ConfirmBy']}\n\n"
        f"Reply YES to confirm or NO if you can't make it. Thank you!"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python send_opportunity.py <OpportunityID>")
        sys.exit(1)

    opportunity_id = sys.argv[1]

    spreadsheet = get_spreadsheet()
    opportunity = get_opportunity(spreadsheet, opportunity_id)
    volunteers = get_volunteers(spreadsheet)

    if not volunteers:
        print("No active volunteers found in the sheet.")
        sys.exit(1)

    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    whatsapp_from = os.environ["TWILIO_WHATSAPP_FROM"]

    sent, failed = 0, 0
    for volunteer in volunteers:
        message = build_message(volunteer["name"], opportunity)
        to_number = volunteer["phone"]
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"

        try:
            client.messages.create(from_=whatsapp_from, to=to_number, body=message)
            sent_at = datetime.now(timezone.utc).isoformat()
            log_message_sent(spreadsheet, opportunity_id, volunteer["name"], volunteer["phone"], sent_at)
            sent += 1
            print(f"Sent to {volunteer['name']} ({volunteer['phone']})")
        except Exception as exc:
            failed += 1
            print(f"Failed to send to {volunteer['name']} ({volunteer['phone']}): {exc}")

    print(f"\nDone. Sent: {sent}  Failed: {failed}")


if __name__ == "__main__":
    main()
