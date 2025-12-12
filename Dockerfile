FROM python:3.12-slim

# Install system deps for Node.js and general utils
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

# Install Claude Code CLI globally (for SDK compatibility)
RUN npm install -g @anthropic-ai/claude-code

# Set working directory
WORKDIR /app

# 4. 关键步骤：只复制 pyproject.toml 和 poetry.lock
# 这一步是缓存优化的核心。只有当这两个文件改变时，才会重新执行后续的 poetry install。
COPY pyproject.toml poetry.toml  poetry.lock ./

# Install Python dependencies with Poetry
RUN poetry install --no-root --sync

# Copy the app code
COPY . .

# --- BEGIN CHANGES ---

# 1. Create a non-root user and group called "appuser"
RUN groupadd -r appuser && useradd -m -g appuser appuser

# 2. Change ownership of the app directory and the Poetry installation
#    The poetry installation is in /root/.local, so we change its ownership too.
RUN chown -R appuser:appuser /app

WORKDIR /workspace

# 3. Switch to the non-root user
USER appuser

# --- END CHANGES ---

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (development mode with reload; switch to --no-reload for prod)
#CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
