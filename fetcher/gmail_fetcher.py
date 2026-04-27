import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class GmailFetcher:
    def __init__(self):
        self.service = self._get_service()
        self.since_date = datetime.strftime(datetime.today() - timedelta(days=7),'%Y-%m-%d')

    def _get_service(self):
        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
        creds = None
        if os.path.exists("fetcher/token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("fetcher/gmail_credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("fetcher/token.json", "w") as token:
                token.write(creds.to_json())

        try:
            # Call the Gmail API
            service = build("gmail", "v1", credentials=creds)
            print("Successfully connected to gmail")
            return service
        except HttpError as error:
            # TODO(developer) - Handle errors from gmail API.
            print(f"An error occurred: {error}")
            return None

    def _decode_part(self,data: str, mime_type: str) -> str:
        """Decode a single base64 encoded email part."""
        try:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8")
            if mime_type == "text/html":
                # Strip HTML tags — get clean readable text
                soup = BeautifulSoup(decoded, "html.parser")
                # Remove noise elements
                for tag in soup(["script", "style", "nav",
                                 "footer", "header", "img"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)
            else:
                # Plain text — return as is
                return decoded
        except Exception as e:
            print(f"Decode error: {e}")
            return ""

    def _get_email_body(self, payload: dict) -> str:
        """
        Extract email body from Gmail payload.
        Handles three email structures:
        1. Simple single part email
        2. Multipart email (most newsletters)
        3. Deeply nested multipart email
        """
        html_body = ""
        plain_body = ""

        def extract_parts(parts: list):
            """Recursively search through all parts."""
            nonlocal html_body, plain_body

            for part in parts:
                mime_type = part.get("mimeType", "")
                data = part.get("body", {}).get("data", "")

                # Nested multipart — go deeper
                if "parts" in part:
                    extract_parts(part["parts"])

                # HTML version — prefer this, richer content
                elif mime_type == "text/html" and data:
                    if not html_body:  # take first one found
                        html_body = self._decode_part(data, "text/html")

                # Plain text version — fallback
                elif mime_type == "text/plain" and data:
                    if not plain_body:
                        plain_body = self._decode_part(data, "text/plain")

        # Case 1 — email has multiple parts (most newsletters)
        if "parts" in payload:
            extract_parts(payload["parts"])

        # Case 2 — simple single part email
        elif "body" in payload:
            data = payload["body"].get("data", "")
            mime_type = payload.get("mimeType", "text/plain")
            if data:
                if mime_type == "text/html":
                    html_body = self._decode_part(data, "text/html")
                else:
                    plain_body = self._decode_part(data, "text/plain")

        # Prefer HTML — it has more content than plain text
        raw_body = html_body or plain_body

        if not raw_body:
            return "No content found in email."

        # Clean up excessive blank lines
        lines = [
            line.strip()
            for line in raw_body.splitlines()
            if line.strip()
        ]
        clean_body = "\n".join(lines)

        # Cap at 8000 chars for LLM token limits
        return clean_body[:8000]

    def _extract_email(self, sender_email: str):
        query = f'{sender_email} after:{self.since_date}'
        try:
            result = self.service.users().messages().list(
                userId="me",
                q=query,
                maxResults=50
            ).execute()

            messages = result.get("messages", [])
            print(f"Found {len(messages)} emails\n")

            emails = []
            for msg in messages:
                email_data = self.service.users().messages().get(
                    userId="me",
                    id=msg["id"],
                    format="full"
                ).execute()
                emails.append(email_data)
            return emails

        except HttpError as error:
            print(f"An error occurred: {error}")

    def fetch(self,sender_email: str):
        emails = self._extract_email(sender_email)
        newsletters = []
        for email in emails:
            # Getting the subject (title) of newspaper
            headers = email['payload'].get('headers', [])
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
            # Retrieving content
            content = self._get_email_body(email['payload'])
            newsletters.append({'title': subject, 'source':sender_email,'content': content})
        return newsletters


if __name__ == "__main__":
    gmail = GmailFetcher().fetch('test_newspaper@gmail.com')
    print(gmail)

