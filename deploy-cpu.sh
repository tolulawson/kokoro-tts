#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting CPU deployment of Kokoro TTS..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for docker compose command (either version)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "📦 Building and starting Docker containers..."

# Navigate to CPU docker directory and start services
cd docker/cpu
$DOCKER_COMPOSE up --build -d

# Check if containers started successfully
if [ $? -eq 0 ]; then
    echo "✅ Deployment successful! The service is now running."
    echo "🌐 API is available at: http://localhost:8880"
    echo "📚 API Documentation: http://localhost:8880/docs"
    echo "🖥️ Web Interface: http://localhost:8880/web"
else
    echo "❌ Deployment failed. Please check the logs above for errors."
    exit 1
fi

# Print how to view logs
echo -e "\n💡 To view logs, run: $DOCKER_COMPOSE logs -f"
echo "💡 To stop the service, run: $DOCKER_COMPOSE down" 