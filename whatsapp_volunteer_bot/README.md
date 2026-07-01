# WhatsApp Volunteer Bot

Sends one-on-one WhatsApp messages to volunteers about a specific volunteering
opportunity (task, confirm-by date, event date), and automatically tracks
their YES/NO replies in a Google Sheet.

- `send_opportunity.py` — sends the opportunity message to every active volunteer.
- `webhook_server.py` — small Flask server that receives replies and marks each
  volunteer Confirmed/Declined/Unclear.
- `sheets_client.py` — shared Google Sheets read/write helpers.

## 1. Set up the Google Sheet

Create a Google Sheet with three tabs (exact header names matter):

**Volunteers**
| Name | Phone | Active |
|------|-------|--------|
| Jordan Lee | +15551234567 | yes |

- `Phone` must be in E.164 format (`+` and country code).
- `Active` can be left blank (treated as yes) or set to `no` to skip someone.

**Opportunities**
| OpportunityID | Task | Details | ConfirmBy | EventDate |
|---------------|------|---------|-----------|-----------|
| beach-cleanup-jul | Beach Cleanup | Bring gloves, we'll provide bags | 2026-07-10 | 2026-07-15 |

- `OpportunityID` is a short slug you pass on the command line when sending.

**Responses** (leave empty — it's populated automatically)
| OpportunityID | Name | Phone | Status | MessageSentAt | RepliedAt | RawReply |
|---------------|------|-------|--------|----------------|-----------|----------|

## 2. Create a Google service account

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a project (or reuse one) and enable the **Google Sheets API**.
2. Create a service account, then create a JSON key for it and download it as `service_account.json` into this folder (do not commit it).
3. Open your Google Sheet and share it with the service account's email address (found in the JSON key, `client_email` field) with **Editor** access.
4. Copy the spreadsheet ID from its URL: `https://docs.google.com/spreadsheets/d/<THIS_PART>/edit`.

## 3. Set up Twilio WhatsApp

1. Sign up at [twilio.com](https://www.twilio.com/) and grab your **Account SID** and **Auth Token** from the console.
2. For testing, join the [Twilio WhatsApp Sandbox](https://www.twilio.com/docs/whatsapp/sandbox) — send the join code from your phone to the sandbox number. Note volunteers must also join the sandbox from their own phones to receive test messages.
3. For production, apply for a dedicated WhatsApp Business Sender in the Twilio console. Business-initiated messages sent outside a 24-hour reply window must use a pre-approved [WhatsApp message template](https://www.twilio.com/docs/whatsapp/tutorial/send-whatsapp-notification-messages-templates) — you'll need to submit the opportunity message wording for approval and reference the template instead of freeform `body` text in `send_opportunity.py`.

## 4. Install & configure

```bash
cd whatsapp_volunteer_bot
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Twilio + Google values
```

## 5. Send an opportunity

```bash
python send_opportunity.py beach-cleanup-jul
```

Each volunteer gets a personalized message and a `Pending` row is added to
the Responses tab.

## 6. Run the reply webhook

```bash
python webhook_server.py
```

Expose it publicly (e.g. `ngrok http 5000` while testing) and set the
public URL + `/webhook/whatsapp` as the "when a message comes in" webhook
for your Twilio WhatsApp sender.

When a volunteer replies YES/NO (or close variants), the bot updates their
most recent `Pending` row to `Confirmed`/`Declined` with a timestamp, and
sends back a short acknowledgement.

**Note:** Twilio signature validation compares against `request.url`. If you
run behind a reverse proxy or tunnel that terminates TLS, make sure Flask
sees the original `https://` scheme and host (e.g. via `ProxyFix`), or
validation will fail.
