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

def identify_target_platform(message):
    try:
        prompt = (
            f"This message: {message} was shared in a Telegram channel known for prompting services that artificially inflate social media metrics "
            f"(such as likes, followers, shares, etc.). Based on the content of the message, identify which particular social media platform it is targeting. "
            f"Please ONLY output your response in the format: {{platform}} and no additional information."
        )

        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )

        output = response['choices'][0]['message']['content']
        print(output)

        # Extract platform
        platform = output.strip('{}')
        return platform
    except Exception as e:
        if "content management policy" in str(e):
            print(f"Skipping row due to content filtering: {e}")
            return None
        print(f"An error occurred: {e}. Retrying after 20 seconds...")
        time.sleep(20)  # Wait for 20 seconds
        return identify_target_platform(message)

def process_platform_csv(input_csv, output_csv, skipped_csv):
    processed_df = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame(columns=['message', 'Platform'])
    skipped_df = pd.read_csv(skipped_csv) if os.path.exists(skipped_csv) else pd.DataFrame(columns=['message', 'Error Count'])

    # Read the input CSV file
    df = pd.read_csv(input_csv)

    processed_messages = set(processed_df['message'])
    skipped_messages = set(skipped_df['message'])

    for index, row in df.dropna(subset=['message']).iterrows():
        message = row['message'].strip()
        if message in processed_messages or message in skipped_messages:
            continue  # Skip already processed or skipped messages

        error_count = skipped_df.loc[skipped_df['message'] == message, 'Error Count'].sum() if message in skipped_messages else 0

        if error_count >= 3:
            print(f"Skipping row {index} permanently due to repeated errors.")
            continue

        try:
            platform = identify_target_platform(message)
            if platform is None:
                # Skip this row due to content filtering
                skipped_df = pd.concat([skipped_df, pd.DataFrame([{ 'message': message, 'Error Count': error_count + 1 }])], ignore_index=True)
                skipped_df.to_csv(skipped_csv, index=False)
                continue

            current_row = pd.DataFrame([{ 'message': message, 'Platform': platform }])
            processed_df = pd.concat([processed_df, current_row], ignore_index=True)
            processed_df.to_csv(output_csv, index=False)
            print(f"Row {index} processed and saved.")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped_df = pd.concat([skipped_df, pd.DataFrame([{ 'message': message, 'Error Count': error_count + 1 }])], ignore_index=True)
            skipped_df.to_csv(skipped_csv, index=False)

        time.sleep(5)

# Define file paths
input_csv = sys.argv[1]
output_csv = 'processed_platforms.csv'
skipped_csv = 'skipped.csv'

# Process the CSV file and write results to the output CSV
process_platform_csv(input_csv, output_csv, skipped_csv)
