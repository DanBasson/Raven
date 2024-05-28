# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /raven_wd

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Copy the current directory contents into the container at /app
COPY . /raven_wd


CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]



