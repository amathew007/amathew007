#!/usr/bin/env python3
"""Flask webhook that receives WhatsApp replies via Twilio and auto-tracks confirmations.

Point your Twilio WhatsApp sender's "when a message comes in" webhook at:
    https://<your-host>/webhook/whatsapp

Run locally for testing with ngrok:
    ngrok http 5000
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, Response, request
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from sheets_client import get_spreadsheet, record_reply

load_dotenv()

app = Flask(__name__)

CONFIRM_WORDS = {"yes", "y", "confirm", "confirmed", "i'm in", "im in"}
DECLINE_WORDS = {"no", "n", "decline", "can't", "cant", "cannot"}


def classify_reply(body):
    normalized = body.strip().lower()
    if normalized in CONFIRM_WORDS or normalized.startswith("yes"):
        return "Confirmed"
    if normalized in DECLINE_WORDS or normalized.startswith("no"):
        return "Declined"
    return "Unclear"


@app.route("/webhook/whatsapp", methods=["POST"])
def whatsapp_webhook():
    validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])
    signature = request.headers.get("X-Twilio-Signature", "")
    if not validator.validate(request.url, request.form, signature):
        return Response(status=403)

    from_number = request.form.get("From", "").replace("whatsapp:", "")
    body = request.form.get("Body", "")

    status = classify_reply(body)
    replied_at = datetime.now(timezone.utc).isoformat()

    spreadsheet = get_spreadsheet()
    updated = record_reply(spreadsheet, from_number, status, replied_at, body)

    reply = MessagingResponse()
    if not updated:
        reply.message(
            "Thanks for your reply! We couldn't match it to a pending opportunity, "
            "but we've noted it."
        )
    elif status == "Confirmed":
        reply.message("Great, you're confirmed! Thank you for volunteering \U0001F64C")
    elif status == "Declined":
        reply.message("No problem, thanks for letting us know. We'll catch you next time!")
    else:
        reply.message("Thanks for your reply! Please respond with YES or NO so we can confirm your spot.")

    return Response(str(reply), mimetype="text/xml")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
