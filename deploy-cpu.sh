#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting CPU deployment of Kokoro TTS..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "📦 Building and starting Docker containers..."

# Navigate to CPU docker directory and start services
cd docker/cpu
docker-compose up --build -d

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
echo -e "\n💡 To view logs, run: docker compose -f docker/cpu/docker-compose.yml logs -f"
echo "💡 To stop the service, run: docker compose -f docker/cpu/docker-compose.yml down" 