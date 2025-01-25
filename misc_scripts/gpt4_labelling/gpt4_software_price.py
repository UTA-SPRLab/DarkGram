import os
import openai
import pandas as pd
import time
import sys

# Azure OpenAI credentials
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  # Use environment variable for security
openai.api_base = "<api_base>"  # Your Azure OpenAI endpoint
openai.api_type = 'azure'
openai.api_version = '2024-02-01'

deployment_name = '<deployment_name>' 

def identify_software_price_and_category(message, filename=None):
    try:
        prompt = (
            f"This message: {message} was shared in a Telegram channel known for distributing software. "
            f"[Optional: The message also included an attachment with the filename {filename}.] "
            f"Based on the content of the message [and the optional filename], find the price of the software. "
            f"If the software involves in-app purchases or subscriptions, determine the subscription cost and use it as the price AND also categorize it as Freemium or Premium. "
            f"Please ALWAYS provide a US dollar value and ONLY output your response as <price><category> and no additional information."
        )

        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        output = response['choices'][0]['message']['content']
        print(output)

        # Extract price and category
        try:
            price, category = output.strip('<>').split('><')
        except ValueError:
            price, category = "Unknown", "Unknown"

        return price, category
    except Exception as e:
        if "content management policy" in str(e):
            print(f"Skipping row due to content filtering: {e}")
            return None, None
        print(f"An error occurred: {e}. Retrying after 20 seconds...")
        time.sleep(20)  # Wait for 20 seconds
        return identify_software_price_and_category(message, filename)

def process_software_csv(input_csv, output_csv, skipped_csv):
    # Load or create processed and skipped DataFrames
    processed_df = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame(columns=['message', 'filename', 'Price', 'Category'])
    skipped_df = pd.read_csv(skipped_csv) if os.path.exists(skipped_csv) else pd.DataFrame(columns=['message', 'filename', 'Error Count'])

    df = pd.read_csv(input_csv)

    processed_messages = set(processed_df['message'])
    skipped_messages = set(skipped_df['message'])

    # Process the rows
    for index, row in df.dropna(subset=['message']).iterrows():
        message = row['message'].strip()
        filename = row.get('filename', None)
        if message in processed_messages or message in skipped_messages:
            continue  # Skip already processed or skipped messages

        error_count = skipped_df.loc[skipped_df['message'] == message, 'Error Count'].sum() if message in skipped_messages else 0

        if error_count >= 3:
            print(f"Skipping row {index} permanently due to repeated errors.")
            continue

        try:
            price, category = identify_software_price_and_category(message, filename)
            if price is None and category is None:
                # Skip this row due to content filtering
                skipped_df = pd.concat([skipped_df, pd.DataFrame([{ 'message': message, 'filename': filename, 'Error Count': error_count + 1 }])], ignore_index=True)
                skipped_df.to_csv(skipped_csv, index=False)
                continue

            current_row = pd.DataFrame([{ 'message': message, 'filename': filename, 'Price': price, 'Category': category }])
            processed_df = pd.concat([processed_df, current_row], ignore_index=True)
            processed_df.to_csv(output_csv, index=False)
            print(f"Row {index} processed and saved.")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped_df = pd.concat([skipped_df, pd.DataFrame([{ 'message': message, 'filename': filename, 'Error Count': error_count + 1 }])], ignore_index=True)
            skipped_df.to_csv(skipped_csv, index=False)

        time.sleep(5)

# Define file paths
input_csv = sys.argv[1]
output_csv = 'software_price.csv'
skipped_csv = 'skipped_software_price.csv'

# Process the CSV file and write results to the output CSV
process_software_csv(input_csv, output_csv, skipped_csv)
