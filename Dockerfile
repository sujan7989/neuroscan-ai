# ================================================================
#  NeuroScan AI — Docker image for Render deployment
#  Node.js 20 LTS, serves static frontend + Express backend
# ================================================================

FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy server package files first (layer cache optimisation)
COPY server/package*.json ./server/

# Install server dependencies
RUN cd server && npm install --omit=dev

# Copy the entire project (frontend + backend + ml data)
COPY . .

# Expose port (Render injects PORT env var at runtime)
EXPOSE 3000

# Health check so Render knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget -qO- http://localhost:${PORT:-3000}/api/health || exit 1

# Start the server
CMD ["node", "server/server.js"]
