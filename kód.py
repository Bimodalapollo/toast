import json
import fitz
import jsonlines
print("Libraries are working!")
import openai

# OpenAI API Key
OPENAI_API_KEY = "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"


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def load_json(json_path):
    """Loads a JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_jsonl_entry(pdf_text, quiz_data):
    """Creates a formatted JSONL entry."""
    return {
        "messages": [
            {"role": "system", "content": "You are an AI that generates revision quizzes based on anatomical texts."},
            {"role": "user", "content": pdf_text},
            {"role": "assistant", "content": json.dumps(quiz_data, indent=4)}
        ]
    }


def generate_jsonl(pdf_path, json_path, output_jsonl):
    """Processes a PDF & JSON pair and creates a JSONL file for fine-tuning."""
    pdf_text = extract_text_from_pdf(pdf_path)
    quiz_data = load_json(json_path)

    # Create JSONL format data
    entry = create_jsonl_entry(pdf_text, quiz_data)

    # Save to JSONL
    with jsonlines.open(output_jsonl, "w") as writer:
        writer.write(entry)

    print(f"Data saved to {output_jsonl}")


# Example usage with uploaded files
pdf_file = "C:\\Voovo\\examples\\Copy of Copy of Copy of Airway.pdf"
json_file = "C:\\Voovo\\examples\\Copy of Airways.json"
output_file = "C:\\Voovo\\examples\\Airways_inputok.json"

generate_jsonl(pdf_file, json_file, output_file)

import openai

openai.api_key = OPENAI_API_KEY

# Upload the training file
file_response = openai.files.create(
    file=open("C:\\Voovo\\examples\\Airways_inputok.json", "rb"),
    purpose="fine-tune"
)

print("File uploaded:", file_response)

# Start fine-tuning
fine_tune_response = openai.fine_tuning.jobs.create(
    training_file=file_response.id,
    model="gpt-4o-mini-2024-07-18"
)

print("Fine-tuning started:", fine_tune_response)

response = openai.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[{"role": "user", "content": "Generate quiz questions for respiratory anatomy"}]
)

print(response.choices[0].message.content)
status = openai.fine_tuning.jobs.retrieve("ftjob-XitpT3yobAWxLdK99SNzDT43")
print(status.fine_tuned_model)







