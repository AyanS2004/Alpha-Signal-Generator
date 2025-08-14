# Frontend Dockerfile
FROM node:20-alpine as build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* /app/
RUN npm install --legacy-peer-deps
COPY frontend /app
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


