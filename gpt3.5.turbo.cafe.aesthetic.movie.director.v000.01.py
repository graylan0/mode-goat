import json
import requests
import random
import sys
import sqlite3
from PIL import Image
import io
import os
import base64
from transformers import pipeline
from moviepy.editor import ImageSequenceClip
import openai

# Read API key from config.json
with open("config.json", "r") as f:
    config = json.load(f)
    openai.api_key = config["openai_api_key"]

class AestheticEvaluator:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="cafeai/cafe_aesthetic")

    def evaluate_aesthetic(self, image_path):
        result = self.pipe(image_path)
        return result[0]['score']

class AdvancedImageGenerator:
    def __init__(self):
        self.aesthetic_evaluator = AestheticEvaluator()
        self.conn = sqlite3.connect('image_db.sqlite')
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, filename TEXT, aesthetic_score REAL, clip_features TEXT, last_words TEXT)''')
        self.c.execute('''CREATE TABLE IF NOT EXISTS storyboards (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, storyboard TEXT)''')
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def generate_images(self, message, duration):
        payload = {
            "prompt": message,
            "steps": 9,
            "seed": random.randrange(sys.maxsize),
            "width": 333,
            "height": 411,
        }
        response = requests.post('http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)
        
        if response.status_code != 200:
            print("Error generating image: ", response.status_code)
            return None

        r = response.json()
        best_image = None

        for i, img_data in enumerate(r['images']):
            image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
            filename = f"{message}_{i}.png"
            image.save(filename)
            
            aesthetic_score = self.aesthetic_evaluator.evaluate_aesthetic(filename)
            self.c.execute("INSERT INTO images VALUES (?, ?, ?)", (message, filename, aesthetic_score))
            self.conn.commit()
            
            if aesthetic_score > 0.7 and best_image is None:
                print(f"High aesthetic score: {filename}, Score: {aesthetic_score}")
                best_image = filename

        return best_image

    def generate_movie(self, message, num_frames=30, duration="30s"):
        frames = []
        frame_folder = "frames"

        # Create the folder if it doesn't exist
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        best_image = self.generate_images(message, duration)
        if best_image is None:
            print("No high aesthetic score image found.")
            return

        for i in range(num_frames):
            payload = {
                "prompt": message,
                "steps": 9,
                "seed": random.randrange(sys.maxsize),
                "width": 333,
                "height": 411,
            }
            response = requests.post('http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)
            
            if response.status_code != 200:
                print("Error generating image: ", response.status_code)
                continue

            r = response.json()

            for img_data in r['images']:
                image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
                filename = f"{message}_{i}.png"
                filepath = os.path.join(frame_folder, filename)
                image.save(filepath)
                frames.append(filepath)

        clip = ImageSequenceClip(frames, fps=24)
        video_filename = f"{message}.mp4"
        clip = clip.set_duration(duration)
        clip.write_videofile(video_filename, codec="libx264")

        return video_filename

    def review_with_gpt(self, aesthetic_score, clip_features):
        context = f"Review this image with an aesthetic score of {aesthetic_score} and CLIP features {clip_features}. Should I approve or reject it?"
        messages = [
            {"role": "system", "content": "You are a movie-making assistant."},
            {"role": "assistant", "content": context}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            max_tokens=50
        )
        decision = response['choices'][0]['message']['content'].strip()
        return decision

if __name__ == "__main__":
    generator = AdvancedImageGenerator()
    duration = input("Enter the duration of the movie (e.g., 1s for 1 second, 1m for 1 minute, 1hr for 1 hour): ")
    best_image = generator.generate_movie("A beautiful sunset_movie", num_frames=30, duration=duration)
    if best_image:
        print(f"The generated movie is {best_image}")
