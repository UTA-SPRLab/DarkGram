import configparser
import json
import csv
import os
from datetime import datetime
import asyncio
import re
import time
import logging

import torch
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, MessageMediaDocument, DocumentAttributeFilename
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

###############################################################################
# LOGGING: everything except specific prints will go here
###############################################################################
logging.basicConfig(
    filename='darkgram.log',
    level=logging.INFO,  
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# MODEL LOADING & PREDICTION
###############################################################################
def load_model(model_path='model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device

def predict(text, model, tokenizer, device, category_names):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities

        confidence_scores = probs.cpu().numpy().flatten()
        predicted_class = confidence_scores.argmax()

    return predicted_class, confidence_scores

###############################################################################
# JSON -> CSV PROCESSING
###############################################################################
def get_nested_value(item, keys):
    """
    Safely traverse nested dictionaries to retrieve a value.
    Example:
        value = get_nested_value(obj, ["someKey", "nestedKey"])
    """
    for key in keys:
        if isinstance(item, dict) and key in item:
            item = item[key]
        else:
            return None
    return item

def process_json_file_to_csv(file_path, csv_folder, channel_id):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    csv_data = []
    for item in data:
        flat_item = {
            "id": get_nested_value(item, ["id"]),
            "date": get_nested_value(item, ["date"]),
            "message": get_nested_value(item, ["message"]),
            "out": get_nested_value(item, ["out"]),
            "mentioned": get_nested_value(item, ["mentioned"]),
            "media_unread": get_nested_value(item, ["media_unread"]),
            "silent": get_nested_value(item, ["silent"]),
            "post": get_nested_value(item, ["post"]),
            "from_scheduled": get_nested_value(item, ["from_scheduled"]),
            "legacy": get_nested_value(item, ["legacy"]),
            "edit_hide": get_nested_value(item, ["edit_hide"]),
            "pinned": get_nested_value(item, ["pinned"]),
            "noforwards": get_nested_value(item, ["noforwards"]),
            "peer_channel": get_nested_value(item, ["peer_id", "channel_id"]),
            "from_id_user": get_nested_value(item, ["from_id", "user_id"]),
            "fwd_from": get_nested_value(item, ["fwd_from"]),
            "via_bot_id": get_nested_value(item, ["via_bot_id"]),
            "reply_to_msg_id": get_nested_value(item, ["reply_to", "reply_to_msg_id"]),
            "reply_to_scheduled": get_nested_value(item, ["reply_to", "reply_to_scheduled"]),
            "forum_topic": get_nested_value(item, ["reply_to", "forum_topic"]),
            "media_photo_id": get_nested_value(item, ["media", "photo", "id"]),
            "reply_markup": get_nested_value(item, ["reply_markup"]),
            "views": get_nested_value(item, ["views"]),
            "forwards": get_nested_value(item, ["forwards"]),
            "replies": get_nested_value(item, ["replies"]),
            "edit_date": get_nested_value(item, ["edit_date"]),
            "post_author": get_nested_value(item, ["post_author"]),
            "grouped_id": get_nested_value(item, ["grouped_id"]),
            "reactions": get_nested_value(item, ["reactions"]),
            "restriction_reason": get_nested_value(item, ["restriction_reason"]),
            "ttl_period": get_nested_value(item, ["ttl_period"]),
        }
        csv_data.append(flat_item)

    if not csv_data:
        logger.info(f"No messages to write for channel_id {channel_id}.")
        return

    csv_file_name = os.path.basename(file_path).replace('.json', '.csv')
    out_dir = os.path.join(csv_folder, str(channel_id))
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, csv_file_name)
    csv_columns = csv_data[0].keys()

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    logger.info(f"CSV file '{csv_file}' created with {len(csv_data)} messages.")

###############################################################################
# MISC UTILS
###############################################################################
class DateTimeEncoder(json.JSONEncoder):
    """
    JSON encoder that gracefully handles datetime and bytes.
    """
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, bytes):
            return list(o)
        return json.JSONEncoder.default(self, o)

###############################################################################
# MAIN 
###############################################################################
# Load model, tokenizer, and device only once
model, tokenizer, device = load_model()
category_names = [
    'Credential Compromise',
    'Blackhat Resources',
    'Artificial Boosting',
    'Benign',
    'Copyright Media',
    'Pirated Software'
]

# Init TG client
config = configparser.ConfigParser()
config.read("config.ini")

with open("configs/tg_auth.json", "r") as f:
    tg_auth = json.load(f)
api_id = tg_auth["api_id"]
api_hash = tg_auth["api_hash"]
phone = tg_auth["phone"]
username = tg_auth["username"]

client = TelegramClient(username, api_id, api_hash)

# Keep track of channels we've scanned
scanned_channels = set()

async def analyze_new_channel(url):
    """
    Fetch the most recent messages (limit=10) of a newly discovered channel,
    store them to JSON, convert JSON -> CSV, handle drops. If malicious_count >= 5,
    we log it to new_channels.csv with majority category. We'll identify channels by 'channel_id'.
    """
    lower_url = url.lower()
    if lower_url in scanned_channels:
        logger.info(f"Skipping scan of duplicate channel at {url}")
        return

    # We print these lines directly, as requested (not logged to file):
    print(f"Looking into new channel: {url}")
    scanned_channels.add(lower_url)
    time.sleep(5) # A small break to respect the API limits

    try:
        entity = await client.get_entity(url)
        # channel_id uniquely identifies the channel in Telethon
        channel_id = entity.id

        history = await client(GetHistoryRequest(
            peer=entity,
            offset_id=0,
            offset_date=None,
            add_offset=0,
            limit=10,
            max_id=0,
            min_id=0,
            hash=0
        ))

        all_messages = []
        drops = []
        malicious_count = 0

        # Track how many times each non-benign category appears
        category_counts = {}

        for message in history.messages:
            # Convert the message to a dictionary for JSON dumping
            message_dict = message.to_dict()
            all_messages.append(message_dict)

            # Classification check
            if message.message:
                predicted_class, _ = predict(message.message, model, tokenizer, device, category_names)
                predicted_cat = category_names[predicted_class]
                if predicted_cat != "Benign":
                    malicious_count += 1
                    category_counts[predicted_cat] = category_counts.get(predicted_cat, 0) + 1

            # Collect "drops" info if there's a document
            if isinstance(message.media, MessageMediaDocument):
                post_id = message.id
                post_url = f"{url}/{post_id}"
                filename = "N/A"
                if message.media.document.attributes:
                    for attr in message.media.document.attributes:
                        if isinstance(attr, DocumentAttributeFilename):
                            filename = attr.file_name
                            break
                drops.append({
                    "channel_id": channel_id,
                    "post_id": post_id,
                    "post_url": post_url,
                    "filename": filename
                })

        # 1. Store messages to JSON file
        json_folder = "json_data"
        os.makedirs(json_folder, exist_ok=True)
        json_file_path = os.path.join(json_folder, f"{channel_id}.json")
        with open(json_file_path, 'w', encoding='utf-8') as jf:
            json.dump(all_messages, jf, cls=DateTimeEncoder, ensure_ascii=False, indent=4)

        logger.info(f"Saved {len(all_messages)} messages to {json_file_path} for channel_id={channel_id}.")

        # 2. Convert JSON -> CSV (in 'csvs/channel_id')
        process_json_file_to_csv(json_file_path, "csvs", channel_id)

        # 3. Save drops info to 'drops/channel_id/drops.csv'
        if drops:
            drops_folder = os.path.join("drops", str(channel_id))
            os.makedirs(drops_folder, exist_ok=True)
            drops_file = os.path.join(drops_folder, 'drops.csv')

            file_exists = os.path.isfile(drops_file)
            with open(drops_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["channel_id", "post_id", "post_url", "filename"])
                for drop_info in drops:
                    writer.writerow([
                        drop_info["channel_id"],
                        drop_info["post_id"],
                        drop_info["post_url"],
                        drop_info["filename"]
                    ])
            logger.info(f"Saved {len(drops)} drops to {drops_file} for channel_id={channel_id}.")

        # 4. If malicious_count >= 5, log it to new_channels.csv with majority category 
        if malicious_count >= 5:
            # Find category with the highest count
            majority_category = max(category_counts, key=category_counts.get)
            print(f"Found malicious channel: {url}")

            # Append to new_channels.csv with columns: [channel_url, channel_id, Category]
            new_channels_file = 'new_channels.csv'
            file_exists = os.path.isfile(new_channels_file)
            with open(new_channels_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['channel_url', 'channel_id', 'Category'])
                writer.writerow([url, channel_id, majority_category])

            logger.info(f"Channel {url} (ID: {channel_id}) is malicious. Category: {majority_category}")

        # Taking a break to be polite to Telegram servers
        await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Error analyzing channel {url}: {e}")

async def main(channel, known_ids):
    """
    Fetch messages from a known channel, look for t.me/ links, analyze them if new.
    """
    # Ensure the client is connected and authorized
    if not client.is_connected():
        await client.connect()
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        try:
            code = input('Enter the code: ')
            await client.sign_in(phone, code)
        except SessionPasswordNeededError:
            password = input('Password: ')
            await client.sign_in(password=password)

    # Resolve the channel entity
    try:
        if channel.isdigit():
            entity = PeerChannel(int(channel))
        else:
            entity = await client.get_entity(channel)
    except Exception as e:
        logger.error(f"Could not get entity for channel: {channel}. Error: {e}")
        return

    try:
        history = await client(GetHistoryRequest(
            peer=entity,
            offset_id=0,
            offset_date=None,
            add_offset=0,
            limit=100,
            max_id=0,
            min_id=0,
            hash=0
        ))
    except Exception as e:
        logger.error(f"GetHistoryRequest failed for channel: {channel}. Error: {e}")
        return

    logger.info(f"Fetched {len(history.messages)} messages from channel: {channel}")

    for message in history.messages:
        if message.id in known_ids:
            continue  # Skip known messages
        known_ids.add(message.id)

        if message.message:
            # Look for potential new t.me links in message text
            telegram_links = re.findall(r'(?:https?://)?t\.me/\S+', message.message)
            for link in telegram_links:
                await analyze_new_channel(link)

async def run():
    known_ids = set()

    # Reading channel list
    channel_list = []
    if os.path.isfile('channel_list.txt'):
        with open('channel_list.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    channel_list.append(line)
    else:
        logger.warning("'channel_list.txt' does not exist or is inaccessible.")
        return

    # Continuously poll the channels in a loop
    while True:
        for channel in channel_list:
            try:
                await main(channel, known_ids)
            except Exception as e:
                logger.error(e)
        # Sleep a bit before next polling cycle
        await asyncio.sleep(10)

###############################################################################
# Program Entry
###############################################################################
with client:
    client.loop.run_until_complete(run())
