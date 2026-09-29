FROM python:3.11-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR=/tmp/fragmented-facts-matplotlib \
    HF_HOME=/cache/huggingface
WORKDIR /app
COPY requirements-common.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip

FROM base AS cpu
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements-common.txt
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY ff.py ./
COPY scripts/docker_verify.py ./scripts/docker_verify.py
COPY project_plan/feasibility ./project_plan/feasibility
RUN python -m pip install --no-deps -e .
CMD ["python", "ff.py", "doctor"]

FROM cpu AS cpu-aya
RUN python -m pip install --no-cache-dir bitsandbytes==0.50.2
CMD ["python", "project_plan/feasibility/aya_cpu_smoke.py", "--help"]

FROM base AS gpu
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir -r requirements-common.txt bitsandbytes==0.47.0
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY ff.py ./
COPY scripts/docker_verify.py ./scripts/docker_verify.py
COPY project_plan/feasibility ./project_plan/feasibility
RUN python -m pip install --no-deps -e .
CMD ["python", "ff.py", "doctor"]
