"""
Used for allowing access to Google account. Run this file to initialize or
refresh authentication. Should be run automatically by GoogleTool.
"""

import os
import pickle
from googleapiclient.discovery import build, Resource
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.transport.requests import Request


ROOT = os.getenv("SMARTHOME_ROOT", default=None)
if ROOT is None:
    raise ValueError("Env variable $SMARTHOME_ROOT is not set up.")

SCOPES = ["https://mail.google.com/", "https://www.googleapis.com/auth/calendar"]


def gcloud_authenticate(force_refresh: bool = False, app: str = "gmail") -> Resource:
    if app == "gmail":
        version = "v1"
    else:  # app == 'calendar'
        version = "v3"
    creds = None
    pkl_path = f"{ROOT}/sage/misc_tools/apis/token.pickle"

    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time

    if os.path.exists(pkl_path) and not force_refresh:
        with open(pkl_path, "rb") as token:
            creds = pickle.load(token)

    # if there are no (valid) credentials availablle, let the user log in.

    if not force_refresh:
        print(f"credentials path exists: {os.path.exists(pkl_path)}")
    else:
        print("refreshing google access token")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                f"{ROOT}/sage/misc_tools/apis/gcloud_credentials.json", SCOPES
            )
            creds = flow.run_console()

        # save the credentials for the next run
        with open(pkl_path, "wb") as token:
            pickle.dump(creds, token)

    return build(app, version, credentials=creds)


if __name__ == "__main__":

    gcloud_authenticate(force_refresh=True)
