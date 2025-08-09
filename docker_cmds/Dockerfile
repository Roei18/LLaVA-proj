# Use Miniconda as base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your code into the image
COPY . .

# Create the environment
RUN conda env create -f llava_yaml.yml

# Activate conda env by default
SHELL ["conda", "run", "-n", "llava_yaml", "/bin/bash", "-c"]

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/llava"

# Make all scripts executable (optional)
RUN chmod +x ./scripts/*

# Default command (overridable at runtime)
CMD ["bash"]
