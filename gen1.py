# Transformer models
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Logging and progress handling
from transformers import logging as transformers_logging


from typing import Any, Dict, List, Tuple

import PIL
import time
import torch
import gc
from tqdm import tqdm

from transformers import pipeline, set_seed
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline




def resize_image(image: PIL.Image.Image, max_width: int, max_height: int) -> PIL.Image.Image:
    """Resize the image to fit within specified dimensions while maintaining aspect ratio."""
    original_width, original_height = image.size

    aspect_ratio = original_width / original_height
    new_width = min(max_width, original_width)
    new_height = int(new_width / aspect_ratio)

    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
    return resized_image

class ImageAnnotationGenerator:
    def __init__(self, model_name: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = None
        self.load_model()

    def load_model(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16,cache_dir='./weight/')
        self.model.to(self.device)

    def clear_gpu(self):
        self.model.to('cpu')
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def generate_description(self,
                             image: PIL.Image.Image,
                             verbose: bool = False,
                             max_new_tokens: int = 20) -> str:
        """
        Generates a string description for a given image by interfacing with a transformer
        model using prompt-based captioning and building conversational context.
    
        Args:
            image (PIL.Image.Image): The image for which the description is to be generated.
            verbose (bool, optional): If True, additional logging information is printed. Defaults to False.
            max_new_tokens (int, optional): The maximum number of tokens to generate for each prompt. Defaults to 20.
    
        Returns:
            str: A generated description of the image.
        """
        if not verbose:
            transformers_logging.set_verbosity_error()

        description = ""
        prompts = ["A picture of", "The setting is", "The background is", "The image type/style is"]
        
        for i, prompt in enumerate(prompts):
            description += prompt + ' '
            inputs = self.processor(image, text=description, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens) #GPT2Tokenizer
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if verbose:
                print(f"{i}. Prompt: {prompt}")

            
            if answer:
                # Remove any ending spaces or punctuation that is not a period
                answer = answer.rstrip(" ,;!?")
                # Add a period at the end if it's not already there
                if not answer.endswith('.'):
                    answer += '.'
                    
                description += answer + ' '
            else:
                description = description[:-len(prompt) - 1]

        if not verbose:
            transformers_logging.set_verbosity_info()
            
        # Remove any trailing spaces and ensure the description ends with a period
        description = description.strip()
        if not description.endswith('.'):
            description += '.'
        return description

    def generate_annotation(
            self,
            image: PIL.Image.Image,
            original_dimensions: tuple,
            resize: bool,
            verbose: int) -> dict:
        """
        Generate a text annotation for a given image.
    
        Parameters:
        image_id (int or str): The identifier for the image within the dataset.
        dataset_name (str): The name of the dataset the image belongs to.
        image (PIL.Image.Image): The image object that requires annotation.
        original_dimensions (tuple): Original dimensions of the image as (width, height).
        resize (bool): Allow image downsizing to maximum dimensions of (1280, 1280).
        verbose (int): Verbosity level.
    
        Returns:
        dict: Dictionary containing the annotation data.
        """
        image_to_process = image.copy()
        if resize: # Downsize if dimension(s) are greater than 1280
            image_to_process = resize_image(image_to_process, 1280, 1280)
            if verbose > 1 and image_to_process.size != image.size:
                print(f"Resized : {image.size} to {image_to_process.size}")
        try:
            description = self.generate_description(image_to_process, verbose > 2)
            annotation = {
                'description': description,
                'original_dimensions': f"{original_dimensions[0]}x{original_dimensions[1]}",
            }
            return annotation
        except Exception as e:
            if verbose > 1:
                bt.logging.error(f"Error processing image in : {e}")
            return None

    def process_image(
            self,
            image_info: dict,

            
            resize: bool,
            verbose: int) -> Tuple[Any, float]:

        if image_info['image'] is None:
            if verbose > 1:
                bt.logging.debug(f"Skipping image in dataset  due to missing image data.")
            return None, 0

        original_dimensions = image_info['image'].size
        start_time = time.time()
        annotation = self.generate_annotation(
                                              image_info['image'],
                                              original_dimensions,
                                              resize,
                                              verbose)
        time_elapsed = time.time() - start_time

        if annotation is None:
            if verbose > 1:
                bt.logging.debug(f"Failed to generate annotation for image in dataset ")
            return None, time_elapsed

        return annotation, time_elapsed
    
import numpy as np
import torch
import random
import time
import re
import gc
import os
import warnings

VALIDATOR_MODEL_META = {
    "prompt_generators": [
        {
            "model": "Gustavosta/MagicPrompt-Stable-Diffusion",
            "tokenizer": "gpt2",
            "device": -1
        }
    ],
    "diffusers": [
        {
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True,
            "pipeline": "StableDiffusionXLPipeline"
        }
    ]
}

PROMPT_TYPES = ('random', 'annotation')

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

DIFFUSER_ARGS = {
    m['path']: {k: v for k, v in m.items() if k != 'path' and k != 'pipeline'}  
    for m in VALIDATOR_MODEL_META['diffusers']
}

DIFFUSER_PIPELINE = {
    m['path']: m['pipeline'] for m in VALIDATOR_MODEL_META['diffusers'] if 'pipeline' in m
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-2.7b-coco"







class SyntheticImageGenerator:

    def __init__(
        self,
        prompt_type='random',
        prompt_generator_name=PROMPT_GENERATOR_NAMES[0],
        diffuser_name=DIFFUSER_NAMES[0],
        use_random_diffuser=False,
        image_cache_dir=None
    ):
        if prompt_type not in PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type '{prompt_type}'. Options are {PROMPT_TYPES}")
        if prompt_generator_name not in PROMPT_GENERATOR_NAMES:
            raise ValueError(f"Invalid prompt generator name '{prompt_generator_name}'. Options are {PROMPT_GENERATOR_NAMES}")
        if not use_random_diffuser and diffuser_name not in DIFFUSER_NAMES:
            raise ValueError(f"Invalid diffuser name '{diffuser_name}'. Options are {DIFFUSER_NAMES}")

        self.use_random_diffuser = use_random_diffuser
        self.prompt_type = prompt_type
        self.prompt_generator_name = prompt_generator_name

        if self.use_random_diffuser and diffuser_name is not None:
            bt.logging.warning("Warning: diffuser_name will be ignored (use_random_diffuser=True)")
            self.diffuser_name = None
        else:
            self.diffuser_name = diffuser_name

        self.image_annotation_generator = None
        if self.prompt_type == 'annotation':
            print(f"Loading image captioning model ({IMAGE_ANNOTATION_MODEL})...")
            self.image_annotation_generator = ImageAnnotationGenerator(model_name=IMAGE_ANNOTATION_MODEL)
            self.image_annotation_generator.load_model()
        else:
            print(f"Loading prompt generation model ({prompt_generator_name})...")
            self.prompt_generator = pipeline(
                'text-generation', **PROMPT_GENERATOR_ARGS[prompt_generator_name])

        self.image_cache_dir = image_cache_dir
        self.load_diffuser(self.diffuser_name)
        if image_cache_dir is not None:
            os.makedirs(self.image_cache_dir, exist_ok=True)

    def generate(self, k: int = 1, real_images=None) -> list:
        """
        Generates k synthetic images. If self.prompt_type is 'annotation', a BLIP2 captioning pipeline is used
        to produce prompts by captioning real images. If self.prompt_type is 'random', an LLM is used to generate
        prompts.

        Args:
            k (int): Number of images to generate.

        Returns:
            list: List of dictionaries containing 'prompt', 'image', and 'id'.
        """
        print("Generating prompts...")
        if self.prompt_type == 'annotation':
            if real_images is None:
                raise ValueError(f"real_images can't be None if self.prompt_type is 'annotation'")
            prompts = [
                self.generate_image_caption(real_images[i])
                for i in range(k)
            ]
        elif self.prompt_type == 'random':
            prompts = [
                self.generate_random_prompt(retry_attempts=10)
                for _ in range(k)
            ]
        else:
            raise NotImplementedError
        
        # if self.use_random_diffuser:
        #     self.load_diffuser('random')
        # else:
        #     self.load_diffuser(self.diffuser_name)
        
        gen_data = []
        for prompt in prompts:
            image_data = self.generate_image(prompt)
            if self.image_cache_dir is not None:
                path = os.path.join(self.image_cache_dir, image_data['id'])
                image_data['image'].save(path)
            gen_data.append(image_data)
            
        # self.clear_gpu()  # remove diffuser from gpu

        return gen_data

    def clear_gpu(self):
        if self.diffuser is not None:
            bt.logging.debug(f"Deleting previous diffuser, freeing memory")
            del self.diffuser
            gc.collect()
            torch.cuda.empty_cache()
            self.diffuser = None

    def load_diffuser(self, diffuser_name) -> None:
        """
        loads a huggingface diffuser model.
        """
        if diffuser_name == 'random':
            diffuser_name = np.random.choice(DIFFUSER_NAMES, 1)[0]
        
        print(f"Loading image generation model ({diffuser_name})...")

        self.diffuser_name = diffuser_name
        pipeline_class = globals()[DIFFUSER_PIPELINE[diffuser_name]]
        print(f"Starting to load the diffuser model: {diffuser_name}...")
        self.diffuser = pipeline_class.from_pretrained(diffuser_name,
                                                       torch_dtype=torch.float16,
                                                       **DIFFUSER_ARGS[diffuser_name],
                                                       add_watermarker=False,
                                                       cache_dir="./weight/")
        self.diffuser.set_progress_bar_config(disable=True)
        self.diffuser.to("cuda")
        print(f"Diffuser model {diffuser_name} loaded successfully.")
        print(f"Loaded {diffuser_name} using {pipeline_class.__name__}.")
        print(f"Loaded {diffuser_name} using {pipeline_class.__name__}.")

    def generate_image_caption(self, image_sample) -> str:
        """

        """
        # self.image_annotation_generator.load_model()
        annotation = self.image_annotation_generator.process_image(
            image_info=image_sample,
            resize=False,
            verbose=0,

        )[0]
        # self.image_annotation_generator.clear_gpu()
        return annotation['description']

    def generate_random_prompt(self, retry_attempts: int = 10) -> str:
        """
        Generates a prompt for image generation.

        Args:
            retry_attempts (int): Number of attempts to generate a valid prompt.

        Returns:
            str: Generated prompt.
        """
        seed = random.randint(100, 1000000)
        set_seed(seed)

        starters = [
            'A photorealistic portrait',
            'A photorealistic image of a person',
            'A photorealistic landscape',
            'A photorealistic scene'
        ]
        quality = [
            'RAW photo', 'subject', '8k uhd',  'soft lighting', 'high quality', 'film grain'
        ]
        device = [
            'Fujifilm XT3', 'iphone', 'canon EOS r8' , 'dslr',
        ]

        for _ in range(retry_attempts):
            starting_text = np.random.choice(starters, 1)[0]
            response = self.prompt_generator(
                starting_text, max_length=(77 - len(starting_text)), num_return_sequences=1, truncation=True)

            prompt = response[0]['generated_text'].strip()
            prompt = re.sub('[^ ]+\.[^ ]+','', prompt)
            prompt = prompt.replace("<", "").replace(">", "")

            # temporary removal of extra context (like "featured on artstation") until we've trained our own prompt generator
            prompt = re.split('[,;]', prompt)[0] + ', '
            prompt += ', '.join(np.random.choice(quality, np.random.randint(len(quality)//2, len(quality))))
            prompt += ', ' + np.random.choice(device, 1)[0]
            if prompt != "":
                return prompt

    def get_tokenizer_with_min_len(self):
        """
        Returns the tokenizer with the smallest maximum token length from the 'diffuser` object.
    
        If a second tokenizer exists, it compares both and returns the one with the smaller 
        maximum token length. Otherwise, it returns the available tokenizer.
        
        Returns:
            tuple: A tuple containing the tokenizer and its maximum token length.
        """
        # Check if a second tokenizer is available in the diffuser
        if self.diffuser.tokenizer_2:
            if self.diffuser.tokenizer.model_max_length > self.diffuser.tokenizer_2.model_max_length:
                return self.diffuser.tokenizer_2, self.diffuser.tokenizer_2.model_max_length
        return self.diffuser.tokenizer, self.diffuser.tokenizer.model_max_length

    def truncate_prompt_if_too_long(self, prompt: str):
        """
        Truncates the input string if it exceeds the maximum token length when tokenized.
    
        Args:
            prompt (str): The text prompt that may need to be truncated.
    
        Returns:
            str: The original prompt if within the token limit; otherwise, a truncated version of the prompt.
        """
        tokenizer, max_token_len = self.get_tokenizer_with_min_len()
        tokens = tokenizer(prompt, verbose=False) # Suppress token max exceeded warnings
        if len(tokens['input_ids']) < max_token_len:
            return prompt
        # Truncate tokens if they exceed the maximum token length, decode the tokens back to a string
        truncated_prompt = tokenizer.decode(token_ids=tokens['input_ids'][:max_token_len-1],
                                            skip_special_tokens=True)
        tokens = tokenizer(truncated_prompt)
        print("Truncated prompt to abide by token limit.")
        return truncated_prompt
    
    def generate_image(self, prompt, name = None, generate_at_target_size = False) -> list:
        """
        Generates an image based on a text prompt, optionally at a specified target size.
    
        Args:
            prompt (str): The text prompt used for generating the image.
            name (str, optional): The id associated with the generated image. Defaults to None.
            generate_at_target_size (bool, optional): If True, generates the image at a specified target size.
    
        Returns:
            list: A dictionary containing the prompt (truncated if over max token length), generated image, and image ID.
        """
        # Generate a unique image name based on current time if not provided
        image_name = name if name else f"{time.time():.0f}.jpg"
        # Check if the prompt is too long
        truncated_prompt = self.truncate_prompt_if_too_long(prompt)
        try:
            if generate_at_target_size:
                #Attempt to generate an image with specified dimensions
                gen_image = self.diffuser(prompt=truncated_prompt, height=(256,256)[0],
                                      width=(256,256)[1]).images[0]
            else:
                #Generate an image using default dimensions supported by the pipeline
                gen_image = self.diffuser(prompt=truncated_prompt).images[0]
        except Exception as e:
            if generate_at_target_size:
                bt.logging.warning(f"Attempt with custom dimensions failed, falling back to default dimensions. Error: {e}")
                try:
                    # Fallback to generating an image without specifying dimensions
                    gen_image = self.diffuser(prompt=truncated_prompt).images[0]
                except Exception as fallback_error:
                    bt.logging.error(f"Failed to generate image with default dimensions after initial failure: {fallback_error}")
                    raise RuntimeError(f"Both attempts to generate image failed: {fallback_error}")
            else:
                bt.logging.warning(f"Image generation error: {e}")
            
        image_data = {
            'prompt': truncated_prompt,
            'image': gen_image,
            'id': image_name
        }
        return image_data
    



import numpy as np
from PIL import Image
from io import BytesIO
import requests
from datasets import load_dataset


def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        image = Image.open(img_data).convert('RGB')
        return image
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None


def sample_random_image_from_dataset(dataset, retries=5) -> dict:
    for _ in range(retries):
        try:
            index = np.random.randint(0, len(dataset))  
            sample = dataset[index]
            
            # Handle image retrieval
            if 'url' in sample:
                image = download_image(sample['url'])
                if image is None:
                    raise ValueError("Invalid image from URL")
                image_id = sample['url']
            elif 'image' in sample:
                if isinstance(sample['image'], Image.Image):
                    image = sample['image']
                elif isinstance(sample['image'], bytes):
                    image = Image.open(BytesIO(sample['image']))
                elif isinstance(sample['image'], dict):  
                    image = Image.fromarray(np.array(sample['image']))
                else:
                    raise NotImplementedError("Unsupported image format")
                image_id = sample.get('name', sample.get('filename', str(index)))
            else:
                raise NotImplementedError("Missing 'image' or 'url' field")
            
            if image is not None:

                return {'image': image, 'id': image_id}

        except Exception as e:
            print(f"Error loading image: {e}. Retrying...")


    print(f"Failed to sample a valid image after {retries} retries.")
    return None



real= [
         "bitmind/open-images-v7", 
         "bitmind/ffhq-256", 
         "bitmind/celeb-a-hq", 
       
    ]

synt=SyntheticImageGenerator(prompt_type='annotation', use_random_diffuser=False,diffuser_name='Corcelio/mobius')
random_source=random.choice(real)

os.makedirs('./weight/',exist_ok=True)
os.makedirs('./data_mobius2/',exist_ok=True)

for i in tqdm(range(10000)):
    dataset=load_dataset(random_source,cache_dir='./weight/')
    image_sample = {'image':sample_random_image_from_dataset(dataset['train'])['image']}
    
    sample = synt.generate(
            k=1, real_images=[image_sample])[0]  
    image=sample['image']

    image.save(f'./data_mobius2/synthetic_{i}.jpg')




             




