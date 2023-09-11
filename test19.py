from fastapi import FastAPI, Path
import asyncio
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama

# Initialize logging to debug level to capture detailed logs
logging.basicConfig(level=logging.DEBUG)

# Create an instance of the FastAPI class
app = FastAPI()

# Initialize the Llama2 model
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, "llama-2-7b.ggmlv3.q8_0.bin")
llm = Llama(model_path=model_path, n_ctx=2000)
executor = ThreadPoolExecutor(max_workers=3)

# Function to trim tokens in a string to fit within a given limit
def trim_tokens(text, max_tokens):
    tokens = text.split()
    while len(tokens) > max_tokens:
        tokens.pop(0)  # Remove the first token
    return ' '.join(tokens)

# Asynchronous function to generate text using the Llama2 model
async def llama_generate_async(prompt):
    loop = asyncio.get_event_loop()
    pre_prompt = "Based on the previous context, generate a concise and relevant continuation. Limit your output to 2-3 sentences."
    full_prompt = f"{pre_prompt} {prompt}"
    trimmed_prompt = trim_tokens(full_prompt, 990)
    
    try:
        output = await loop.run_in_executor(executor, lambda: llm(trimmed_prompt, max_tokens=499))
        return output['choices'][0]['text']
    except ValueError as e:
        logging.error(f"Token limit exceeded: {e}")
        return "Token limit exceeded"

# Function to construct the initial prompt for the Multiverse Movie Generator Game
async def construct_initial_prompt(topic):
    rules_prompt = f"Create a writing story prompt to start a Multiverse Movie Generator Game about {topic}."
    initial_prompt = await llama_generate_async(rules_prompt)
    return initial_prompt

# Function to extract key topics from a frame
def extract_key_topics(frame):
    return ' '.join(frame.split()[-3:])

# Function to continue the next frame generation
async def continue_next_frame_generation(last_three_frames):
    # Extract key topics from the last three frames
    key_topics = [extract_key_topics(frame) for frame in last_three_frames]
    combined_key_topics = ' '.join(key_topics)
    
    # Generate a new scene description based on the key topics
    rules_prompt = ("As an AI specialized in Advanced Space Movies, you are tasked with generating a scene description. "
                    f"Generate a scene based on these key topics: {combined_key_topics}")
    new_frame_generation = await llama_generate_async(rules_prompt)
    
    return new_frame_generation

async def generate_advanced_space_scene():
    rules_prompt = ("As an AI specialized in Advanced Space Movies, you are tasked with generating a scene description. "
                    "1. Stay in character as a specialized AI for Advanced Space Movies. "
                    "2. Generate an 18-word description of an advanced space scene.")
    scene_output = await llama_generate_async(rules_prompt)
    return scene_output

# FastAPI endpoint to start the Multiverse Movie Generator Game
@app.get("/movie/{topic}", tags=["movie"])
async def start_movie(topic: str = Path(..., description="The topic of the movie")):
    try:
        # Initial prompt with rules
        initial_prompt = await construct_initial_prompt(topic)
        
        frames = {}
        frames["frame_0"] = initial_prompt
        last_three_frames = [initial_prompt, "", ""]
        
        for i in range(1, 500):
            # Generate advanced space movie scene description
            advanced_space_scene = await generate_advanced_space_scene()
            
            # Continue next frame generation
            new_frame_generation = await continue_next_frame_generation([advanced_space_scene, last_three_frames[-1]])
            
            frames[f"frame_{i}"] = new_frame_generation
            last_three_frames.pop(0)
            last_three_frames.append(new_frame_generation)

            sanitized_topic = ''.join(e for e in topic if e.isalnum())
            sanitized_topic = sanitized_topic[:50]
            
            with open(f"{sanitized_topic}_movie_frames.json", "w") as f:
                json.dump(frames, f, indent=4)
        
        return {"message": f"Advanced Space Movie about {topic} started and 11 frames generated. Saved to {sanitized_topic}_movie_frames.json"}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"message": "An error occurred during movie generation"}

# Main function to run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
