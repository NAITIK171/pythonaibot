# Add python3-dev to the list of packages to install
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*
