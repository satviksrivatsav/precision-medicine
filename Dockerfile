# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- THIS IS THE KEY ---
# Copy the entire backend directory into the container's /app directory
COPY ./backend /app

# Expose port 8080 to the outside world
EXPOSE 8080

# Command to run the uvicorn server when the container launches
# It will look for an 'api.py' file inside the /app directory
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]