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

# --- Data Loading and Preprocessing ---

# Load the dataset
try:
    df = pd.read_csv('hadiths_data.csv')
    logging.info("Successfully loaded 'hadiths_data.csv'") # add log
except FileNotFoundError:
    logging.error("Error: 'hadiths_data.csv' not found.  Please ensure the file exists.")
    exit()
except Exception as e:
    logging.error(f"Error loading 'hadiths_data.csv': {e}")
    exit()


# Clean and preprocess the 'Hadiths Text' column
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

# Optionally, sample the data to reduce the number of data points
try:
    df = df.groupby('Rawi').apply(
        lambda x: x.sample(n=85, random_state=1)
    ).reset_index(drop=True)
    logging.info("Successfully sampled data by 'Rawi'.")
except KeyError as e:
    logging.warning(f"Column 'Rawi' not found. Sampling by 'Rawi' will be skipped.")
except Exception as e:
    logging.warning(f"Error sampling data by 'Rawi': {e}. Sampling step will be skipped.")
    # Continue without sampling if there's an error
    pass  # or consider setting `df` to original data:  df = pd.read_csv('hadiths_data.csv')

# --- Gemini API Setup ---

# Retrieve Google AI API Key from environment variables
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("Error: GEMINI_API_KEY environment variable not set.")
    exit() # exit to not continue the code if there are no API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-1.5-pro" if available
    logging.info("Successfully loaded Gemini model.")
except Exception as e:
    logging.error(f"Error loading Gemini model: {e}")
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
        list: A list of generated questions.  Returns error messages if generation fails.
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

# Set batch size and initialize question list
batch_size = 7  # Adjust batch size based on API limits and performance
questions = [""] * len(df) # Initialize list with empty strings to match the dataframe length
checkpoint_interval = 2  # Save progress every X batches

for i in range(0, len(df), batch_size):
    # Create a batch of hadiths and indices
    batch_hadiths = df['Hadiths Text'][i:i + batch_size].tolist()
    batch_indices = list(range(i, min(i + batch_size, len(df))))

    # Generate questions for the batch
    batch_questions = generate_questions_batch(batch_hadiths, batch_indices)

    # Update the questions list with the generated questions
    for idx, question in zip(batch_indices, batch_questions):
        questions[idx] = question

    # Introduce a random sleep interval to be kind to the API
    sleep_duration = np.random.uniform(8, 10)  # Sleep for a random duration between 8 and 10 seconds
    print(f"Sleeping for {sleep_duration:.2f} seconds...") # debug
    time.sleep(sleep_duration)

    # Save progress periodically
    if (i // batch_size) % checkpoint_interval == 0 and i > 0:  # Save every checkpoint_interval batches, skip the very first iteration
        df['Sample Question'] = questions
        try:
            df.to_csv('hadiths_data_with_questions_checkpoint.csv', index=False)
            logging.info(f"Checkpoint saved at batch {i // batch_size}. Rows processed: {i}") # add log
        except Exception as e:
             logging.error(f"Error saving checkpoint at batch {i // batch_size}: {e}")

# Save final results
df['Sample Question'] = questions
try:
    df.to_csv('hadiths_data_with_questions_final.csv', index=False)
    logging.info("Final results saved to 'hadiths_data_with_questions_final.csv'") # add log
except Exception as e:
    logging.error(f"Error saving final results: {e}")


print("Question generation complete.") # add confirmation print statement