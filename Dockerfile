# docker cheatsheet:
#   docker build --no-cache -t fastar .
#   docker run --rm -it fastar bash

# Select base image. bookworm was chosen because it is the Debian version in
# which Ubuntu 24.04 is based
FROM python:3.10.12-bookworm

# Install the Python depedencies
RUN pip install --no-cache-dir \
    astropy==6.1.7 \
    astroquery==0.4.8 \
    build==1.2.2.post1 \
    flake8==4.0.1 \
    flake8-quotes==3.4.0 \
    flax==0.10.5 \
    h5py==3.12.1 \
    jax==0.6.0 \
    matplotlib==3.10.1 \
    numpy==2.2.5 \
    pylint==2.12.2 \
    pytest==8.3.4 \
    ruff==0.12.3 \
    setuptools==65.5.1 \
    setuptools_scm==8.3.1 \
    Sphinx==4.3.2 \
    sphinx-rtd-theme==1.3.0 \
    tqdm==4.67.1 \
    wheel==0.41.1

# Set the working directory
WORKDIR /root

# Copy the files from the repository to the workdir
COPY . fastar/
