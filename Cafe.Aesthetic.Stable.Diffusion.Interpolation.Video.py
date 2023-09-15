import os
import random
import requests
import sys
import sqlite3
from PIL import Image
import io
import shutil
import base64
from transformers import pipeline

class AestheticEvaluator:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="cafeai/cafe_aesthetic")

    def evaluate_aesthetic(self, image_path):
        result = self.pipe(image_path)
        return result[0]['score']

class AdvancedImageGenerator:
    def __init__(self):
        self.aesthetic_evaluator = AestheticEvaluator()
        current_directory = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_directory, 'image_db.sqlite')
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS images (prompt TEXT, filename TEXT, aesthetic_score REAL)''')
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def generate_images(self, message, num_images=5, output_directory=None):
        generated_images = []
        if output_directory is None:
            output_directory = "output_images_" + str(random.randint(1, 1000))  # Randomized output directory

        os.makedirs(output_directory, exist_ok=True)

        for _ in range(num_images):
            url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
            payload = {
                "prompt": message,
                "steps": 9,
                "seed": random.randrange(sys.maxsize),
                "width": 333,
                "height": 411,
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                r = response.json()
                for i, img_data in enumerate(r['images']):
                    image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
                    random_suffix = str(random.randint(1, 1000))  # Randomized filename suffix
                    filename = os.path.join(output_directory, f"{message}_{i}_{random_suffix}.png")
                    image.save(filename)

                    aesthetic_score = self.aesthetic_evaluator.evaluate_aesthetic(filename)
                    self.c.execute("INSERT INTO images VALUES (?, ?, ?)", (message, filename, aesthetic_score))
                    self.conn.commit()

                    if aesthetic_score > 0.7:
                        print(f"High aesthetic score: {filename}, Score: {aesthetic_score}")
                        generated_images.append(filename)
            else:
                print("Error generating image: ", response.status_code)

        return generated_images


    def generate_best_aesthetic_image(self, message, num_attempts=5):
        best_image = None
        best_score = 0
        for _ in range(num_attempts):
            generated_images = self.generate_images(message)
            for filename in generated_images:
                self.c.execute("SELECT aesthetic_score FROM images WHERE filename=?", (filename,))
                score = self.c.fetchone()[0]
                if score > best_score:
                    best_score = score
                    best_image = filename
        return best_image, best_score

    def interpolate_images(self, initial_message, final_message, output_folder, num_frames=50):
        os.makedirs(output_folder, exist_ok=True)

        initial_image, _ = self.generate_best_aesthetic_image(initial_message)
        final_image, _ = self.generate_best_aesthetic_image(final_message)

        if not initial_image or not final_image:
            print("Failed to generate initial or final images for interpolation.")
            return

        for frame_number in range(num_frames):
            ratio = frame_number / (num_frames - 1)
            blended_image = Image.blend(Image.open(initial_image), Image.open(final_image), ratio)
            filename = os.path.join(output_folder, f"interpolation_{frame_number:03d}.png")
            blended_image.save(filename)
        print(f"Generated {num_frames} interpolation frames in {output_folder}")

def generate_intermediate_frames(self, image_path, output_folder, num_frames=50):
    os.makedirs(output_folder, exist_ok=True)

    if num_frames == 1:
        # Special case when num_frames is 1
        ratio = 0.5  # You can choose any ratio you want
        filename = os.path.join(output_folder, "intermediate_000.png")

        url = 'http://127.0.0.1:7860/sdapi/v1/img2img'
        options = {
            'init_images': [image_path],
            'ratio': ratio  # Adjust the transformation ratio
        }
        response = requests.post(url, json=options)
        if 'images' in response.json():
            img_data = response.json()['images'][0]
            image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
            image.save(filename)
        else:
            print(f"No images received from img2img API for frame 0")
    else:
        for frame_number in range(num_frames):
            ratio = frame_number / (num_frames - 1)
            filename = os.path.join(output_folder, f"intermediate_{frame_number:03d}.png")

            url = 'http://127.0.0.1:7860/sdapi/v1/img2img'
            options = {
                'init_images': [image_path],
                'ratio': ratio  # Adjust the transformation ratio
            }
            response = requests.post(url, json=options)
            if 'images' in response.json():
                img_data = response.json()['images'][0]
                image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
                image.save(filename)
            else:
                print(f"No images received from img2img API for frame {frame_number}")

if __name__ == "__main__":
    generator = AdvancedImageGenerator()
    
    # Generate 3 images using the same seed
    image1 = generator.generate_best_aesthetic_image("A beautiful sunset")[0]
    image2 = generator.generate_best_aesthetic_image("A beautiful sunrise")[0]
    image3 = generator.generate_best_aesthetic_image("A beautiful night sky")[0]

    # Interpolate between these images to create a few frames
    generator.interpolate_images("A beautiful sunset", "A beautiful sunrise", "output_folder1", num_frames=200)
    generator.interpolate_images("A beautiful sunrise", "A beautiful night sky", "output_folder2", num_frames=200)

    # Copy over a couple of frames (e.g., last frame from first interpolation and first frame from second interpolation)
    shutil.copy("output_folder1/interpolation_199.png", "output_folder2/")
    shutil.copy("output_folder2/interpolation_000.png", "output_folder1/")

    # Apply img2img transformations for each frame in the interpolation
    for frame_number in range(200):
        generator.generate_intermediate_frames(image1, f"intermediate_frames_{frame_number:03d}", num_frames=1)

    generator.close()



