# Frontend Dockerfile
FROM node:20-alpine as build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* /app/
RUN npm install --legacy-peer-deps
COPY frontend /app
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/build /usr/share/nginx/html

# Add reverse proxy config for /api to backend service
RUN printf '\nserver {\n  listen 80;\n  root /usr/share/nginx/html;\n  index index.html;\n  location / {\n    try_files $uri /index.html;\n  }\n  location /api/ {\n    proxy_pass http://backend:5000;\n    proxy_http_version 1.1;\n    proxy_set_header Host $host;\n    proxy_set_header X-Real-IP $remote_addr;\n    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n    proxy_set_header X-Forwarded-Proto $scheme;\n  }\n}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


