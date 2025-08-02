# Use a Python base image that includes common build tools
# We'll use Python 3.9 as it's often stable for many libraries.
# You can try 3.10 or 3.11 if 3.9 causes issues with specific library versions.
FROM python:3.9-slim-buster

# Install system dependencies required for scientific libraries
# These are crucial for compiling packages like numpy, pandas, scipy, scikit-learn, etc.
# build-essential: Provides compilers (gcc, g++)
# libblas-dev, liblapack-dev: Linear algebra libraries often needed by numpy/scipy
# gfortran: Fortran compiler, sometimes needed by scipy
# pkg-config: Helper for finding libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container and install Python dependencies
# Using --no-cache-dir to prevent caching package data, which saves space
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Command to run your application when the container starts
# This will execute your main.py script
CMD ["python", "main.py"]
