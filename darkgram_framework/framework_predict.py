import configparser
import json
import csv
import os
from datetime import datetime
import asyncio
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, MessageMediaDocument
from tqdm import tqdm
import glob
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

def load_model(model_path='model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device

# Predict the category and confidence scores for a given post message
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

# Custom JSON encoder to handle datetime and bytes
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, bytes):
            return list(o)
        return json.JSONEncoder.default(self, o)

def get_nested_value(item, keys):
    for key in keys:
        if isinstance(item, dict) and key in item:
            item = item[key]
        else:
            return None
    return item

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

api_id = # Add api_id
api_hash = # Add api_hash
phone = # Add phone number
username = #Add username

client = TelegramClient(username, api_id, api_hash)

async def main(channel, known_ids, highest_message_ids):

    # Ensure the client is connected and authorized
    if not client.is_connected():
        await client.connect()
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone, input('Enter the code: '))
        except SessionPasswordNeededError:
            await client.sign_in(password=input('Password: '))

    # Resolve the channel entity
    if channel.isdigit():
        entity = PeerChannel(int(channel))
    else:
        entity = await client.get_entity(channel)

    channel_name = entity.title if hasattr(entity, 'title') else 'channel'

    base_dir = 'raw' # Raw metadata
    channel_dir = os.path.join(base_dir, channel_name)
    os.makedirs(channel_dir, exist_ok=True)

    drops_base_dir = 'drops' # Attachments
    drops_channel_dir = os.path.join(drops_base_dir, channel_name)
    os.makedirs(drops_channel_dir, exist_ok=True)

    # Find the latest message ID from CSV
    csv_folder_path = f'csv/{channel_name}/'
    csv_folder_save_path = f'csv/'
    os.makedirs(csv_folder_path, exist_ok=True)
    latest_file = max(glob.glob(f'{csv_folder_path}/*.csv'), default=None, key=os.path.getctime)
    file = open(f'{csv_folder_path}/url.txt', "w")
    file.write(f'{channel}')
    file.close()

    last_collected_id = 0
    if latest_file:
        with open(latest_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                last_collected_id = int(row['id'])
                break

    min_id = last_collected_id + 1 if last_collected_id else 0

    # Fetch messages newer than the last collected ID
    all_messages = []
    total_count_limit = 100  # Modify as needed

    history = await client(GetHistoryRequest(
        peer=entity,
        offset_id=0,
        offset_date=None,
        add_offset=0,
        limit=total_count_limit,
        max_id=0,
        min_id=min_id,
        hash=0
    ))

    if not history.messages:
        print(f"No new messages to collect from {channel_name}.")
        time.sleep(2)
        return
    else:
        print(f"Will collect new data from {channel_name}")
        print("Small break before starting collection..")
        time.sleep(20)

    for message in history.messages:
        if message.id in known_ids:
            continue  # Skip known messages
        known_ids.add(message.id)
        all_messages.append(message.to_dict())
        highest_message_ids[channel_name] = max(highest_message_ids.get(channel_name, 0), message.id)

        # Handle document media
        if isinstance(message.media, MessageMediaDocument):
            file_name = None
            for attribute in message.media.document.attributes:
                if hasattr(attribute, 'file_name'):
                    file_name = attribute.file_name
                    break

            if file_name and (file_name.endswith('.txt') or file_name.endswith('.csv')):
                time_stamp_dir = os.path.join(drops_channel_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                os.makedirs(time_stamp_dir, exist_ok=True)
                file_path = os.path.join(time_stamp_dir, file_name)
                await client.download_media(message.media, file_path)
                time.sleep(5)  # Rest between downloading media

    # Save messages to JSON
    if all_messages:
        filename = f'{datetime.now().timestamp()}.json'
        json_file_path = os.path.join(channel_dir, filename)
        with open(json_file_path, 'w') as outfile:
            json.dump(all_messages, outfile, cls=DateTimeEncoder)
        process_json_file_to_csv(json_file_path, csv_folder_save_path, channel_name)


def process_json_file_to_csv(file_path, csv_folder, channel_name):
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

    if csv_data:
        csv_file_name = os.path.basename(file_path).replace('.json', '.csv')
        csv_file = os.path.join(csv_folder, channel_name, csv_file_name)
        os.makedirs(os.path.join(csv_folder, channel_name), exist_ok=True)
        csv_columns = csv_data[0].keys()
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        print(f"CSV file '{csv_file}' created.")

    # After processing the CSV, run predictions on the 'message' column and save the results
    prediction_file = 'prediction.csv'
    with open(prediction_file, 'a', newline='', encoding='utf-8') as pred_file:
        pred_writer = csv.writer(pred_file)
        if os.stat(prediction_file).st_size == 0:
            # Write headers if the file is empty
            pred_writer.writerow(['channel_name', 'post_id', 'message', 'prediction', 'confidence_scores'])

        for row in csv_data:
            message = row.get("message", "")
            post_id = row.get("id", "")
            if message:
                predicted_class, confidence_scores = predict(message, model, tokenizer, device, category_names)
                confidence_str = ', '.join([f"{category_names[i]}: {confidence_scores[i]:.4f}" for i in range(len(category_names))])
                pred_writer.writerow([channel_name, post_id, message, category_names[predicted_class], confidence_str])

        print(f"Predictions for channel '{channel_name}' saved to '{prediction_file}'.")


async def run():
    known_ids = set()
    highest_message_ids = {}

    # Reading channel list from a file
    channel_list = []
    with open('channel_list.txt', 'r') as file:
        for line in file:
            channel_list.append(line.strip())  # Remove any trailing newline characters

    channel_counter = 0  

    while True:  # Loop indefinitely
        for channel in channel_list:
            try:
                await main(channel, known_ids, highest_message_ids)
                channel_counter += 1  # Increment the counter each time a channel is processed

                if channel_counter == 30:  # Check if 30 channels have been processed
                    print("Taking a 10-minute break...")
                    await asyncio.sleep(600)  # Sleep for 10 minutes (600 seconds)
                    channel_counter = 0  # Reset the counter

            except Exception as e:
                print(e)


with client:
    client.loop.run_until_complete(run())
