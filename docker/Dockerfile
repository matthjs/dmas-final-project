FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies for rendering graphics
RUN apt-get install -y xvfb x11-utils

# Copy the entire project code to the container
COPY . .

ENV PYTHONPATH="/app:/app/dmas_final_project"

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

RUN poetry lock
# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Run experiment for results
# To get single run graphs run manually.
CMD ["bash", "dmas_final_project/bash.sh"]