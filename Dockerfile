# Use conda-forge base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

COPY . .

# Create conda environment
RUN conda env create -f environment.yml

# Add conda environment to PATH
ENV PATH /opt/conda/envs/food_classifier/bin:$PATH

# Activate the environment
RUN /bin/bash -c "source activate food_classifier"

# Expose the port the app runs on
EXPOSE 5000

# Allow the image to execute the script
RUN chmod +x /app/dist/run.sh

# Set the default command
ENTRYPOINT [".app/dist/run.sh"]