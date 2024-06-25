# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY startup.py ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set the Hugging Face API token as an environment variable
# You can replace 'your_huggingface_token' with your actual token or use a build argument
ENV HUGGINGFACE_TOKEN=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

CMD ["python", "startup.py"]