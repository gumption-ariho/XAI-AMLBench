# Build context = ./frontend
FROM node:20-alpine

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
# Rewrites are baked in at build time; "backend" is the compose service name.
ENV BACKEND_URL=http://backend:8000

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
