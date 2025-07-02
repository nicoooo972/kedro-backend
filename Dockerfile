ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

WORKDIR /home/kedro_docker

# Copy the project configuration and source
COPY pyproject.toml .
COPY src/ src/
COPY conf/ conf/

# Install the project and its dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["kedro"]

# Set the default command
CMD ["run"]
