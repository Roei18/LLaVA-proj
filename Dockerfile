# Use Miniconda as base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your code into the image
COPY . .

# Create the environment
RUN conda env create -f environment.yml

# Activate conda env by default
SHELL ["conda", "run", "-n", "llava_yaml", "/bin/bash", "-c"]

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/llava"

RUN chmod +x ./scripts/* && chmod +x ./scripts/v1_5/*

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "llava_yaml"]
# Default command (overridable at runtime)
CMD ["bash"]
