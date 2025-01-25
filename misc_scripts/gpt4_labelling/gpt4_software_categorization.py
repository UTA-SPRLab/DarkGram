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

# categories
categories = [
    "Utility", "Entertainment", "Communication", "Media & Video", "Photo & Video", 
    "VPN", "Hacking", "Photography", "Music & Audio", "Productivity", "Social Media",
    "Gaming", "Education", "Health & Fitness", "Finance"
]

def identify_software_category_and_description(message, filename=None):
    try:
        prompt = (
            f"This message: {message} was shared in a Telegram channel known for distributing software. "
            f"[Optional: The message also included an attachment with the filename {filename}.] "
            f"Based on the content of the message [optional: and the provided filename], please assign the referenced software "
            f"to one of the following 15 categories: {', '.join(categories)}. If it does not fit into any of these predefined "
            f"categories, feel free to propose a new one. After categorizing the software, kindly provide a brief description "
            f"of its functionality or purpose. Please ONLY output your response as: {{Software_name, Category, Description of Software}} "
            f"and no additional information."
        )

        # Use Azure OpenAI to process the prompt
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        output = response['choices'][0]['message']['content']
        print(output)

        # Extract software name, category, and description
        try:
            software_name, category, description = [item.strip() for item in output.strip('{}').split(',')]
        except ValueError:
            software_name, category, description = "Unknown", "Unknown", "Parsing Error"

        return software_name, category, description
    except Exception as e:
        if "content management policy" in str(e):
            print(f"Skipping row due to content filtering: {e}")
            return None, None, None
        print(f"An error occurred: {e}. Retrying after 20 seconds...")
        time.sleep(20)  # Wait for 20 seconds
        return identify_software_category_and_description(message, filename)

def process_software_csv(input_csv, output_csv, skipped_csv):
    processed_df = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame(columns=['message', 'filename', 'Software Name', 'Category', 'Description'])
    skipped_df = pd.read_csv(skipped_csv) if os.path.exists(skipped_csv) else pd.DataFrame(columns=['message', 'filename', 'Error Count'])

    df = pd.read_csv(input_csv)

    # Track already processed messages
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
            software_name, category, description = identify_software_category_and_description(message, filename)
            if software_name is None and category is None and description is None:
                # Skip this row due to content filtering
                skipped_df = pd.concat([skipped_df, pd.DataFrame([{ 'message': message, 'filename': filename, 'Error Count': error_count + 1 }])], ignore_index=True)
                skipped_df.to_csv(skipped_csv, index=False)
                continue

            current_row = pd.DataFrame([{ 'message': message, 'filename': filename, 'Software Name': software_name, 'Category': category, 'Description': description }])
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
output_csv = 'software_category.csv'
skipped_csv = 'skipped_category.csv'

process_software_csv(input_csv, output_csv, skipped_csv)
