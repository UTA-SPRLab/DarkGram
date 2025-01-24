import os
import csv
import json
import asyncio
from datetime import datetime
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import PeerChannel
import time

def load_telegram_credentials(config_path="configs/tg_creds.json"):
    """
    Loads Telegram credentials from config
    """
    try:
        with open(config_path, "r") as config_file:
            creds = json.load(config_file)
            required_keys = {"api_id", "api_hash", "phone", "username"}
            if not required_keys.issubset(creds.keys()):
                raise ValueError(f"Missing required keys in {config_path}. Expected: {required_keys}")
            return creds
    except FileNotFoundError:
        print(f"Telegram credentials file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON format in Telegram credentials file: {config_path}")
        raise

async def download_drops():
    """
    Reads drops.csv files in the 'drops' folder and downloads the specified files.
    """
    # Load TG credentials
    creds = load_telegram_credentials()
    api_id = creds["api_id"]
    api_hash = creds["api_hash"]
    phone = creds["phone"]
    username = creds["username"]

    client = TelegramClient(username, api_id, api_hash)

    try:
        # Make sure the client is connected and authorized
        if not client.is_connected():
            await client.connect()
        if not await client.is_user_authorized():
            await client.send_code_request(phone)
            try:
                await client.sign_in(phone, input("Enter the code: "))
            except SessionPasswordNeededError:
                await client.sign_in(password=input("Enter your password: "))

        # Base directories
        drops_base_dir = "drops"
        downloaded_base_dir = "downloaded/drops"

        # Iterate over subdirectories in the drops folder
        for channel_id in os.listdir(drops_base_dir):
            channel_dir = os.path.join(drops_base_dir, channel_id)
            if not os.path.isdir(channel_dir):
                continue

            drops_csv_path = os.path.join(channel_dir, "drops.csv")
            if not os.path.exists(drops_csv_path):
                print(f"No drops.csv found for channel {channel_id}. Skipping.")
                continue

            # Create corresponding downloaded folder
            channel_download_dir = os.path.join(downloaded_base_dir, channel_id)
            os.makedirs(channel_download_dir, exist_ok=True)

            # Read drops.csv and process each entry
            with open(drops_csv_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    post_id = row.get("post_id")
                    post_url = row.get("post_url")
                    filename = row.get("filename")

                    if not post_id or not post_url or not filename:
                        print(f"Invalid entry in {drops_csv_path}: {row}. Skipping.")
                        continue

                    # Generate path to save the downloaded file
                    file_save_path = os.path.join(channel_download_dir, filename)

                    if os.path.exists(file_save_path):
                        print(f"File already downloaded: {file_save_path}. Skipping.")
                        continue

                    try:
                        # Download the media 
                        print(f"Downloading {filename} from {post_url}...")
                        message = await client.get_messages(PeerChannel(int(channel_id)), ids=int(post_id))
                        await client.download_media(message.media, file_save_path)
                        print(f"File saved to {file_save_path}.")
                        time.sleep(2)  # Be polite to Telegram servers
                    except Exception as e:
                        print(f"Failed to download {filename} from {post_url}: {e}")

    except Exception as e:
        print(f"Error in download_drops: {e}")

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(download_drops())
