import json
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from gtts import gTTS
import os
from IPython.display import Audio, display

# Load a summarization model from Hugging Face
summarizer = pipeline("summarization", model="Chandans01/custom-chandan-samsum")

# Load the JSON data from a file
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print("Loaded JSON data successfully.")
            return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        raise

# Extract and combine prompts for a specific video
def combine_prompts(video_data):
    try:
        prompts = set(item["prompt"] for item in video_data["data"])
        combined = " ".join(prompts)
        return combined
    except Exception as e:
        print(f"Error combining prompts: {e}")
        raise

# Remove duplicate sentences from the combined text
def remove_duplicate_sentences(text):
    sentences = text.split('. ')
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    return '. '.join(unique_sentences) + '.'

# Generate a detailed description from combined prompts
def generate_detailed_description(text):
    try:
        summary = summarizer(text, max_length=250, min_length=100, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error generating description"

# Save descriptions to a PDF file with text wrapping
def save_to_pdf(video_descriptions, output_pdf_path):
    try:
        pdf = SimpleDocTemplate(output_pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        styleN = styles['BodyText']
        styleN.alignment = TA_JUSTIFY
        elements = []

        for video_path, description in video_descriptions:
            elements.append(Paragraph(f"<b>Video Path:</b> {video_path}", styleN))
            elements.append(Paragraph(f"<b>Description:</b> {description}", styleN))
            elements.append(Paragraph("=" * 80, styleN))

        pdf.build(elements)
        print(f"Descriptions saved to {output_pdf_path}")
    except Exception as e:
        print(f"Error saving to PDF: {e}")

# Generate audio files from descriptions
def save_to_audio(video_descriptions, output_audio_dir):
    audio_file_paths = []
    try:
        if not os.path.exists(output_audio_dir):
            os.makedirs(output_audio_dir)

        for video_path, description in video_descriptions:
            # Create audio file name based on video path
            audio_file_name = os.path.join(output_audio_dir, f"{os.path.basename(video_path)}.mp3")
            tts = gTTS(text=description, lang='en')
            tts.save(audio_file_name)
            audio_file_paths.append(audio_file_name)  # Collect audio file paths
            print(f"Audio saved to {audio_file_name}")
    except Exception as e:
        print(f"Error saving to audio: {e}")
    
    return audio_file_paths  # Return the list of audio file paths

# Main function to load, combine, remove duplicates, and generate descriptions for all videos
def main(file_path, output_pdf_path, output_audio_dir):
    try:
        json_data = load_json_file(file_path)
        video_descriptions = []
        for video in json_data["data"]:
            video_path = video["video_path"]
            combined_prompts = combine_prompts(video)
            unique_prompts = remove_duplicate_sentences(combined_prompts)
            description = generate_detailed_description(unique_prompts)
            video_descriptions.append((video_path, description))

            print(f"Video Path: {video_path}")
            print(f"Summary: {description}")
            print("\n" + "=" * 80 + "\n")

        save_to_pdf(video_descriptions, output_pdf_path)
        audio_file_paths = save_to_audio(video_descriptions, output_audio_dir)
        
        # Play audio for each description
        for audio_file_path in audio_file_paths:
            display(Audio(audio_file_path, autoplay=False))  # Play audio live in the notebook

    except Exception as e:
        print(f"Error in main function: {e}")

# Example usage
file_path = '/kaggle/working/train_data/caption.json'  # Replace with your JSON file path
output_pdf_path = '/kaggle/working/video_summaries.pdf'  # Path where the PDF will be saved
output_audio_dir = '/kaggle/working/audio_summaries'  # Directory to save audio files
main(file_path, output_pdf_path, output_audio_dir)
