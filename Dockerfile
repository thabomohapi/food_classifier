# Use conda-forge base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "food_classifier", "/bin/bash", "-c"]

# Copy the application code
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# Create necessary directories
RUN mkdir -p checkpoints/

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production
ENV CUDA_VISIBLE_DEVICES=0

# Expose the port the app runs on
EXPOSE 5000

# Initialize conda in bash
RUN conda init bash

# Set the entrypoint to use conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "food_classifier"]

# Set the default command
CMD ["python", "-m", "src.app"]