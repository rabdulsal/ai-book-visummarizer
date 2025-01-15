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
'''
 Fetch a summary for the book from a free online resource.
    In this case, we use OpenAI's API to generate a brief summary.
    
'''
def fetch_book_summary(book_title):
    
    # Prompt for summarizing the book into structured sections
    prompt = (
        f"Provide a short and structured summary for the book '{book_title}' by breaking it into the following sections:\n\n"
        "1. Problem: What problem does this book address?\n"
        "2. Solution: What is the overall solution or insight the book offers?\n"
        "3. Principle 1: What is one key principle or idea shared in the book?\n"
        "4. Principle 2: What is another key principle or idea shared in the book?\n"
        "5. Outcome: What are the potential benefits or outcomes for someone who applies the book's teachings?\n\n"
        "Format the response as a JSON object with keys 'problem', 'solution', 'principle_1', 'principle_2', and 'outcome'."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-4 if you have access
            messages=[
                {"role": "system", "content": """
                You are an assistant specialized in creating highly engaging book summaries 
                designed for display in short-form social media style.
                """
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        # Parse the JSON response
        summary_text = response["choices"][0]["message"]["content"]
        summary_dict = eval(summary_text)  # Convert the JSON string to a dictionary

        return summary_dict

    except openai.error.OpenAIError as e:
        print(f"Error fetching book summary: {e}")
        return {
            "problem": "Could not fetch problem description.",
            "solution": "Could not fetch solution description.",
            "principle_1": "Could not fetch first principle.",
            "principle_2": "Could not fetch second principle.",
            "outcome": "Could not fetch outcome.",
        }

# OLD
def generate_summary(book_title):
    """Generate a summary using ChatGPT."""
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Summarize the book in one sentence."},
                  {"role": "user", "content": f"Book Title: {book_title}"}]
    )
    return response['choices'][0]['message']['content']

# Takes a summary sentence and generates and image-creation prompt
def generate_image_prompt(summary_sentence):
    """
    Generate a prompt for image creation based on the summary sentence.
    """
    prompt_template = """
    Acting as an expert social media content-creator and short-form video creator, 
    generate prompts that would be best for creating 3-5 images that 
    tell an eye-catching and engaging visual story representing the following book summary: '{}'
    The generated
    """
    return prompt_template.format(summary_sentence)

def generate_image_prompts(book_title, summary):
    """
    Generate visually engaging prompts for AI image generation, based on the book summary.
    """
    # Breaking the summary into a short narrative arc
    problem_prompt = (
        f"A photorealistic image of a young man struggling with {summary['problem']}, depicted alone in a crowded or dynamic setting. "
        f"Surrounding people are blurred, while the central figure seems isolated or thoughtful. "
        f"Lighting and atmosphere should convey both struggle and potential for change."
    )
    solution_intro_prompt = (
        f"A cinematic shot of a young man reading a book, his face depicting inspiration and hope."
        f"He is in a well-lit library with people scattered about, blurred in the background."
    )
    key_principle_1_prompt = (
        f"A cinematic, high-definition photo of two people engaging in a warm conversation in a work lounge area, symbolizing {summary['principle_1']}. "
        f"The listener leans forward with genuine interest, and their body language conveys empathy and attentiveness. "
        f"A vibrant, sun-lit backdrop with soft-blurring highlight connection."
    )
    key_principle_2_prompt = (
        f"A digital painting of workplace meeting where a male leader is gesturing toward a female colleague to show sincere appreciation, symbolizing {summary['principle_2']}. "
        f"Expressions of gratitude and positive collaboration fill the scene. The atmosphere well-lit, cheerful and professional, with slight blur  on background colleagues."
    )
    outcome_prompt = (
        f"A cinematic shot of confident woman, centered and surrounded by a diverse group work colleagues in an office-setting. Everyone smiling and interacting warmly. "
        f"The scene is lively and colorful, symbolizing {summary['outcome']}. Background is well-lit should exude happiness and success."
    )

    # Returning the prompts as a dictionary
    return {
        "problem": problem_prompt,
        "solution_intro": solution_intro_prompt,
        "principle_1": key_principle_1_prompt,
        "principle_2": key_principle_2_prompt,
        "outcome": outcome_prompt,
    }


# OLD
def generate_visual(prompt, output_path):
    """Generate visuals using DALL·E."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"prompt": prompt, "n": 1, "size": "1024x1024"}
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
    image_url = response.json()["data"][0]["url"]
    image_data = requests.get(image_url).content
    with open(output_path, "wb") as img_file:
        img_file.write(image_data)

from pathlib import Path

# Loops through each summary sentence and generates an appropriate prompt and images
def generate_images(book_title):
    
    """
    Generate images based on the structured dictionary data from the book summary.
    This function uses the generate_image_prompts function to create DALL-E prompts 
    and generates corresponding images for each part of the summary.
    """
    summary_dict = fetch_book_summary(book_title)

    # Directory to save the images
    output_dir = Path("static/generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompts based on the summary dictionary
    prompts = generate_image_prompts(book_title, summary_dict)

    # Dictionary to store image paths and prompts for display on the web page
    image_data = {}

    # Height and width for Tik Tok vids / Reels
    width = 720
    height = 1280

    # Loop through the prompts and generate images
    for key, prompt in prompts.items():
        try:
            # Generate image using OpenAI's API
            # response = openai.Image.create(
            #     prompt=prompt,
            #     n=1,
            #     size="1024x1024",
            # )

            # Save the image locally
            # image_url = response["data"][0]["url"]
            image_url = f"https://pollinations.ai/p/{prompt}?width={width}&height={height}"
            image_path = output_dir / f"{book_title.replace(' ', '_')}_{key}.png"
            with open(image_path, "wb") as f:
                f.write(requests.get(image_url).content)

            # Store the image data
            image_data[key] = {
                "prompt": prompt,
                "image_path": str(image_path)
            }

        except openai.error.OpenAIError as e:
            print(f"Error generating image for {key}: {e}")
            image_data[key] = {
                "prompt": prompt,
                "image_path": None
            }

    return image_data

#OLD
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

# OLD
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

# OLD
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
    # book_title = request.form['book_title']
    # visual_prompt = request.form['visual_prompt']
    
    # # Paths for output files
    # image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'visual.png')
    # audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'voiceover.mp3')
    # video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_video.mp4')

    # # Step 1: Generate summary
    # summary = generate_summary(book_title)

    # # Step 2: Generate visual
    # generate_visual(visual_prompt, image_path)

    # # Step 3: Generate voiceover
    # generate_voiceover(summary, audio_path)

    # # Step 4: Create video
    # create_video(image_path, audio_path, video_path)

    # # Step 5: Post to social media
    # # post_to_social_media(video_path)

    # # return redirect(url_for('index', success=True))

    # # Send video path to the webpage for display
    # return render_template('index.html', video_path=video_path)

    book_title = request.form.get("book_title")
    image_data = generate_images(book_title)
    return render_template("results.html", book_title=book_title, image_data=image_data)

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
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
    app.run(debug=True)
