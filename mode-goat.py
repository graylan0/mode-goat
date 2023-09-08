from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from moviepy.editor import ImageSequenceClip
from PIL import Image
import pytesseract
import asyncio
import random
import requests
import asyncio
import sys
import datetime
import io
import base64
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import uvicorn
# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI
app = FastAPI()

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

# Initialize Lock for thread safety
seed_pool_lock = Lock()

# Initialize Llama2
from llama_cpp import Llama
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, "llama-2-7b.ggmlv3.q8_0.bin")
llm = Llama(model_path=model_path, n_ctx=100)

# Prompt Engineering
context_info = "Generate a Story Frame By Frame with as Llama2Stable LLM2IMG"
multiple_prompts = ["What happens next? this this frame"]
seed_pool = [random.randint(0, 10000) for _ in range(10)]

def check_token_count(text):
    return len(text.split())

# Function to truncate or summarize text
def truncate_or_summarize(text):
    return " ".join(text.split()[:25])

async def llama_generate_async(prompt, max_tokens=100):
    try:
        if len(prompt.split()) > 99:
            prompt = " ".join(prompt.split()[99])
        
        if check_token_count(prompt) > max_tokens:
            prompt = truncate_or_summarize(prompt)
        
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, lambda: llm(prompt, max_tokens=min(max_tokens, 200)))
        
        if output is not None:
            generated_text = output.get('choices', [{}])[0].get('text', '')
            print(f"Llama Generated Text: {generated_text}")  # This will print the generated text
            return generated_text
        else:
            return None
    except Exception as e:
        if "too many tokens" in str(e):
            logging.error("Text too long, truncating...")
            return await llama_generate_async(truncate_or_summarize(prompt), max_tokens)
        else:
            logging.error(f"Error in llama_generate_async: {e}")
            return None
    


async def chunk_and_generate(prompt, max_tokens=50, max_length=72, frame_number=None):
    generated_text = ""
    word_count = 0
    MAX_TOKEN_LIMIT = 50

    while True:
        remaining_tokens = MAX_TOKEN_LIMIT - len(prompt.split())
        if remaining_tokens <= 0:
            print("Error: Prompt too long, skipping this chunk.")
            break

        effective_max_tokens = min(max_tokens, remaining_tokens)
        segment_story = await llama_generate_async(prompt, effective_max_tokens)
        
        if segment_story is not None:
            generated_text += segment_story
            word_count += len(segment_story.split())
            if word_count >= max_length:
                break

    return generated_text


async def generate_images(prompt: str, prev_seed: int):
    images = []
    seed = prev_seed if prev_seed else random.randrange(sys.maxsize)
    url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
    payload = {
        "prompt": prompt,
        "steps": 3,
        "seed": seed,
        "enable_hr": "false",
        "denoising_strength": "0.7",
        "cfg_scale": "7",
        "width": 390,
        "height": 219,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        try:
            r = response.json()
            for i in r['images']:
                images.append(Image.open(io.BytesIO(base64.b64decode(i))))
        except ValueError as e:
            logging.error(f"Error processing image data: {e}")
    else:
        logging.error(f"Error generating image: {response.status_code}")
    return images, seed


@app.get("/generate_movie/{topic}")
async def generate_movie(topic: str):
    try:
        print('Starting movie generation...')
        storyline = ""
        total_frames = 50
        SOME_MAX_LENGTH = 72

        with seed_pool_lock:
            prev_seed = seed_pool.pop(0)

        # Create a folder for the movie
        movie_folder = os.path.join("movies", topic)
        image_folder = os.path.join(movie_folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        for frame in range(total_frames):
            print(f"Processing frame {frame}...")
                       
            text_prompt = f"{context_info} Generate Story Frame for Frame Number: {frame} for Topic: {topic}\n"
            
            generated_text = await chunk_and_generate(text_prompt, max_tokens=200, max_length=SOME_MAX_LENGTH)
            
            if generated_text:
                storyline += generated_text

            # Generate and save images for this frame
            images, new_seed = await generate_images(generated_text, prev_seed)
            if images:
                image_path = os.path.join(image_folder, f"{frame}_{topic}.png")
                images[0].save(image_path)  # Save the first image in the list
                prev_seed = new_seed  # Update the seed for the next iteration

        # After saving all the frames, generate the movie (This should be outside the loop)
        image_files = [os.path.join(image_folder, f"{i}_{topic}.png") for i in range(total_frames)]
        clip = ImageSequenceClip(image_files, fps=24)  # Adjust fps as needed
        movie_path = os.path.join(movie_folder, f"{topic}.mp4")
        clip.write_videofile(movie_path)

        return JSONResponse(content={"message": "Movie generated successfully!", "storyline": storyline, "movie_path": movie_path})

    except Exception as e:
        logging.error(f"Error in generate_movie: {e}")
        return JSONResponse(content={"message": "An error occurred while generating the movie.", "error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
