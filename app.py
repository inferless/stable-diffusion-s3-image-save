import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
import boto3
import uuid


AWS_ACCESS_KEY_ID = os.environ.get('AWS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.environ.get('REGION_NAME')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

class InferlessPythonModel:
    def initialize(self):
        model_id = "stabilityai/stable-diffusion-2-1"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_safetensors=True,
            torch_dtype=torch.float16).to("cuda"
        )

    def infer(self, inputs):
        prompt = inputs["prompt"]
        image = self.pipe(prompt).images[0]
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID , aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=REGION_NAME )
        buff = BytesIO()
        bucket_name = BUCKET_NAME
        key_name = f'{uuid.uuid4()}.jpg'
        image.save(buff, format="JPEG")
        buff.seek(0)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key_name,
            Body=buff,
            ContentType='image/jpeg'
        )
        
        return { "s3_loc" : BUCKET_NAME+key_name }
        
    def finalize(self):
        self.pipe = None
