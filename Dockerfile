FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY tests ./tests
COPY data_sample ./data_sample

RUN pip install --no-cache-dir -r requirements.txt

CMD ["pytest", "-q"]
