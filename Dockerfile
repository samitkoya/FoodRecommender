FROM python:3.10

WORKDIR /code

# Copy the requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy all your project files
COPY . /code

# RUN THE PIPELINE DURING THE BUILD
# IMPORTANT: You must add GEMINI_API_KEY to your Hugging Face space secrets for this to work
RUN python scripts/run_pipeline.py

# Hugging Face Spaces expect web servers to run on port 7860
CMD ["uvicorn", "src.inference_service:app", "--host", "0.0.0.0", "--port", "7860"]
