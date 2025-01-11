from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import openai  # ChatGPT and DALL·E API
import requests  # For IFTTT webhook and DALL·E requests
import ffmpeg  # For video editing
from google.cloud import texttospeech  # Google TTS

from dotenv import load_dotenv
load_dotenv()

import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_CLOUD_CREDENTIALS = os.getenv('GOOGLE_CLOUD_KEY_PATH')
IFTTT_WEBHOOK_URL = f"https://maker.ifttt.com/trigger/upload_video/with/key/{os.getenv('IFTTT_WEBHOOK_KEY')}"


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/abdulsar/Desktop'  # Save videos on the desktop

# Ensure output folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Helper Functions ----------
def generate_summary(book_title):
    """Generate a summary using ChatGPT."""
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Summarize the book in one sentence."},
                  {"role": "user", "content": f"Book Title: {book_title}"}]
    )
    return response['choices'][0]['message']['content']

def generate_visual(prompt, output_path):
    """Generate visuals using DALL·E."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"prompt": prompt, "n": 1, "size": "1024x1024"}
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
    image_url = response.json()["data"][0]["url"]
    image_data = requests.get(image_url).content
    with open(output_path, "wb") as img_file:
        img_file.write(image_data)

def generate_voiceover(text, output_path):
    """Generate AI voiceover using Google Cloud TTS."""
    from google.oauth2 import service_account
    from google.cloud.texttospeech import TextToSpeechClient

    credentials = service_account.Credentials.from_service_account_file(GOOGLE_CLOUD_CREDENTIALS)
    client = TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as audio_file:
        audio_file.write(response.audio_content)

def create_video(image_path, audio_path, video_path):
    """Combine visuals and voiceover into a video."""
    try:
        ffmpeg.input(image_path, loop=1, t=6).output(
            video_path, vcodec="libx264", pix_fmt="yuv420p", acodec="aac", audio=audio_path
        ).run()
    except ffmpeg.Error as e:
        stdout = e.stdout.decode("utf8") if e.stdout else "No stdout output"
        stderr = e.stderr.decode("utf8") if e.stderr else "No stderr output"
        print("stdout:", stdout)
        print("stderr:", stderr)
        raise

def post_to_social_media(video_path):
    """Post video to social media using IFTTT webhook."""
    payload = {"value1": video_path, "value2": "AI-generated content", "value3": "#SelfImprovement #AIArt"}
    requests.post(IFTTT_WEBHOOK_URL, json=payload)

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect user inputs
    book_title = request.form['book_title']
    visual_prompt = request.form['visual_prompt']
    
    # Paths for output files
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'visual.png')
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'voiceover.mp3')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_video.mp4')

    # Step 1: Generate summary
    summary = generate_summary(book_title)

    # Step 2: Generate visual
    generate_visual(visual_prompt, image_path)

    # Step 3: Generate voiceover
    generate_voiceover(summary, audio_path)

    # Step 4: Create video
    create_video(image_path, audio_path, video_path)

    # Step 5: Post to social media
    # post_to_social_media(video_path)

    # return redirect(url_for('index', success=True))

    # Send video path to the webpage for display
    return render_template('index.html', video_path=video_path)

@app.route('/publish', methods=['POST'])
def publish():
    # Handle file selection and upload
    if 'file' not in request.files:
        return redirect(url_for('index', error="No file selected"))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error="No file selected"))
    
    # Validate file type (only video files allowed)
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        return redirect(url_for('index', error="Invalid file type"))

    # Save the uploaded file temporarily
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)

    # Call social media publishing function
    post_to_social_media(upload_path)  # Predefined function
    return redirect(url_for('index', success="Video successfully published!"))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
