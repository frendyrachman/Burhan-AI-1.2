import numpy as np
import google.generativeai as genai
import pandas as pd
import os
import time
import logging

# --- Configuration ---

# Configure logging
logging.basicConfig(
    filename='generation_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- API Key Setup ---

# Retrieve Google AI API Key from environment variables
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("Error: GEMINI_API_KEY environment variable not set.")
    exit() # exit to not continue the code if there are no API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Loading ---

# Load the Gemini model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-1.5-pro" if available
    logging.info("Successfully loaded Gemini model.")
except Exception as e:
    logging.error(f"Error loading Gemini model: {e}")
    exit()

# --- Data Loading and Preprocessing ---

# Load the dataset
try:
    df = pd.read_csv('hadiths_data.csv')
    logging.info("Successfully loaded 'hadiths_data.csv'")
except FileNotFoundError:
    logging.error("Error: 'hadiths_data.csv' not found. Please ensure the file exists.")
    exit()
except Exception as e:
    logging.error(f"Error loading 'hadiths_data.csv': {e}")
    exit()

# Ensure 'Sample Question' column exists. Create if missing and fill with empty string
if 'Sample Question' not in df.columns:
    df['Sample Question'] = ""
    logging.warning("Column 'Sample Question' not found. Created and initialized with empty strings.")

# Clean and preprocess the 'Hadiths Text' column (If not already preprocessed)
# This part is optional, if the cleaning already done
try:
    df["Hadiths Text"] = df["Hadiths Text"].astype(str).apply(
        lambda x: " ".join(x.replace("\n", " ")
                             .replace("\t", " ")
                             .replace("\r", " ")
                             .split()))
    logging.info("Successfully preprocessed 'Hadiths Text' column.")
except KeyError as e:
    logging.error(f"Error: Missing column '{e}' in the dataset. Please check the column names.")
    exit()
except Exception as e:
    logging.error(f"Error preprocessing 'Hadiths Text' column: {e}")
    exit()

# --- Question Generation Function ---

def generate_questions_batch(hadiths, indices, retries=3):
    """
    Generates questions for a batch of hadiths using the Gemini API.

    Args:
        hadiths (list): A list of hadith texts.
        indices (list): A list of indices corresponding to the hadiths in the original DataFrame.
        retries (int): The number of retries if the API call fails.

    Returns:
        list: A list of generated questions. Returns error messages if generation fails.
    """
    response_texts = []
    for hadith, index in zip(hadiths, indices): # Use zip for iteration
        attempt = 0
        while attempt < retries:
            try:
                prompt = f"""Generate a question in interrogative form, based on the following hadith as the answer:\n\n{hadith}\n\n"""
                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                    max_output_tokens=128,
                    temperature=0.8
                ))

                generated_text = response.text.strip() if response else "Error: No response"
                print(f"\n[DEBUG] Row {index}") # debugging
                print(f"[DEBUG] Generated Question: {generated_text}\n") # debugging
                response_texts.append(generated_text)
                logging.info(f"Row {index} - Generated question: {generated_text[:50]}...") # add log
                break  # If successful, break the retry loop
            except Exception as e:
                attempt += 1
                error_message = str(e)
                logging.error(f"Row {index} - Attempt {attempt} - Error: {error_message}")
                if attempt == retries:
                    response_texts.append(f"Error after {retries} attempts: {error_message}")
                    break  # Break the loop if retries are exhausted
                else:
                    time.sleep(5)  # Wait before retrying

    return response_texts

# --- Main Processing Loop ---

# Set batch size and other parameters
batch_size = 7  # Adjust based on API limits and performance
checkpoint_interval = 2  # Save progress every X batches

# Identify rows with null values in 'Sample Question'
null_indices = df[df['Sample Question'].isnull() | (df['Sample Question'] == "")].index.tolist()  # Include empty strings too

# Start processing only if there are rows to process
if null_indices:
    print(f"Generating questions for {len(null_indices)} hadiths...")
    for i in range(0, len(null_indices), batch_size):
        # Create a batch of hadiths and indices
        batch_indices = null_indices[i:i + batch_size]
        batch_hadiths = df.loc[batch_indices, 'Hadiths Text'].tolist()

        # Generate questions for the batch
        batch_questions = generate_questions_batch(batch_hadiths, batch_indices)

        # Update the 'Sample Question' column with generated questions
        for idx, question in zip(batch_indices, batch_questions):
            df.loc[idx, 'Sample Question'] = question

        # Introduce a random sleep interval
        sleep_duration = np.random.uniform(9, 10) # Sleep a bit longer
        print(f"Sleeping for {sleep_duration:.2f} seconds...") # debug
        time.sleep(sleep_duration)

        # Save progress periodically
        if (i // batch_size) % checkpoint_interval == 0 and i > 0:  # Skip first iteration
            try:
                df.to_csv('hadiths_data_with_questions_checkpoint.csv', index=False)
                logging.info(f"Checkpoint saved at batch {i // batch_size}. Rows processed: {i}") # add log
            except Exception as e:
                logging.error(f"Error saving checkpoint at batch {i // batch_size}: {e}")
else:
    print("No rows with missing questions found. Skipping question generation.")
    logging.info("No rows with missing questions found. Skipping question generation.")
# --- Save Results ---
try:
    df.to_csv('hadiths_data_with_questions_final.csv', index=False)
    logging.info("Final results saved to 'hadiths_data_with_questions_final.csv'") # add log
except Exception as e:
    logging.error(f"Error saving final results: {e}")
# --- Display Result (Optional) ---
print("\n=== Final DataFrame ===")
print(df) # Print the dataframe

print("Question generation complete.")