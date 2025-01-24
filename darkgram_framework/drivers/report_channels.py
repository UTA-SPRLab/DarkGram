import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import csv
import json
import os
import time

# Custom message based on the category of the CAC
CATEGORY_MESSAGES = {
    "Credential Compromise": "This channel appears to be sharing content related to credential compromise.",
    "Blackhat Resources": "This channel seems to distribute blackhat hacking resources.",
    "Artificial Boosting": "This channel seems to promote artificial boosting techniques.",
    "Copyright Media": "This channel seems to share copyrighted media without authorization.",
    "Pirated Software": "This channel appears to distribute pirated software."
}

def load_email_config(config_path="configs/email_config.json"):
    """
    Loads email server credentials from the email_config.json file.
    """
    try:
        with open(config_path, "r") as config_file:
            email_config = json.load(config_file)
            required_keys = {"smtp_server", "smtp_port", "email_address", "email_password"}
            if not required_keys.issubset(email_config.keys()):
                raise ValueError(f"Missing required keys in {config_path}. Expected: {required_keys}")
            return email_config
    except FileNotFoundError:
        print(f"Email configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON format in email configuration file: {config_path}")
        raise

def attach_file_to_email(message, file_path):

    try:
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(file_path)}"
        )
        message.attach(part)
    except Exception as e:
        print(f"Failed to attach file {file_path}: {e}")

def send_email(email_config, subject, body, attachment_path=None):
    """
    Sends an email to Telegram Abuse
    """
    recipient_email = "abuse@telegram.org"
    try:
        # Create the email
        message = MIMEMultipart()
        message["From"] = email_config["email_address"]
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        # Attach channel post CSV if available
        if attachment_path:
            attach_file_to_email(message, attachment_path)

        with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
            server.starttls()
            server.login(email_config["email_address"], email_config["email_password"])
            server.send_message(message)
        print(f"Email sent to {recipient_email}.")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

def process_and_send_emails():
    while True:  # retry every 10 minutes
        try:
            # Load email configuration
            email_config = load_email_config()

            # Check if new_channels.csv exists and has unreported channels
            if not os.path.exists("new_channels.csv"):
                print("new_channels.csv not found. Waiting for 10 minutes.")
                time.sleep(600)
                continue

            unreported_channels = []
            with open("new_channels.csv", "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get("Reported") == "False":
                        unreported_channels.append(row)

            if not unreported_channels:
                print("No unreported channels found. Waiting for 10 minutes.")
                time.sleep(600)
                continue

            # Process each unreported channel
            for row in unreported_channels:
                channel_url = row.get("channel_url")
                category = row.get("Category")
                channel_id = row.get("channel_id")

                if not channel_url or not category or not channel_id:
                    print("Invalid entry in new_channels.csv, skipping.")
                    continue

                category_message = CATEGORY_MESSAGES.get(category, "This channel has been flagged for review.")
                email_body = (
                    f"Hello,\n\n"
                    f"We are reporting the following Telegram channel:\n\n"
                    f"Channel URL: {channel_url}\n"
                    f"Category: {category}\n\n"
                    f"Reason: {category_message}\n\n"
                    f"We have also attached a small sample of posts from the channel for your review.\n\n"
                    f"Best,\n"
                    f"DarkGram reporting"
                )

                attachment_path = os.path.join("csvs", f"{channel_id}.csv")
                if not os.path.isfile(attachment_path):
                    print(f"CSV file not found for channel ID {channel_id}, skipping email.")
                    continue

                # Send the email
                send_email(email_config, f"Report: Telegram Channel - {category}", email_body, attachment_path)

                # Mark the channel as reported
                row["Reported"] = "True"

            # Update new_channels.csv with the reported status
            with open("new_channels.csv", "w", newline="") as csvfile:
                fieldnames = ["channel_url", "Category", "channel_id", "Reported"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(unreported_channels)

        except Exception as e:
            print(f"Error processing and sending emails: {e}")

        # Wait for 10 minutes before retrying
        print("Waiting for 10 minutes before retrying...")
        time.sleep(600)

if __name__ == "__main__":
    process_and_send_emails()
