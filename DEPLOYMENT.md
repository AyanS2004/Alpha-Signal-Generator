# 🚀 Alpha Signal Engine - Free Deployment Guide

This guide provides step-by-step instructions for deploying your Alpha Signal Engine to the internet for free using various cloud platforms.

## 📋 Prerequisites

- GitHub account
- Git installed locally
- Your Alpha Signal project pushed to GitHub

## 🎯 Deployment Options

### Option 1: Vercel + Railway (Recommended)

**Best for**: Production-ready deployment with excellent performance

#### Frontend (Vercel)
1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "New Project"
   - Import your repository
3. **Configure Build Settings**:
   - Framework Preset: `Create React App`
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `build`
4. **Environment Variables**:
   - Add `REACT_APP_API_URL` = `https://your-railway-backend-url.railway.app`
5. **Deploy**: Click "Deploy"

#### Backend (Railway)
1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
2. **Configure Service**:
   - Railway will auto-detect Python
   - Set Start Command: `python backend/app.py`
   - Set Port: `5000`
3. **Environment Variables** (optional):
   - `ALPACA_API_KEY` = your_api_key
   - `ALPACA_SECRET_KEY` = your_secret_key
4. **Deploy**: Railway will automatically deploy

**Cost**: Free (Vercel: 100GB bandwidth/month, Railway: $5 credit/month)

---

### Option 2: Render (All-in-One)

**Best for**: Simple deployment with everything in one place

#### Deploy to Render
1. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New" → "Blueprint"
   - Connect your repository
2. **Configure Services**:
   - Render will detect the `render.yaml` file
   - It will create both frontend and backend services
3. **Environment Variables** (optional):
   - Add your Alpaca API keys in the backend service
4. **Deploy**: Render will automatically deploy both services

**Cost**: Free (750 hours/month, sleeps after 15min inactivity)

---

### Option 3: Netlify + Railway

**Best for**: Alternative to Vercel with similar performance

#### Frontend (Netlify)
1. **Connect to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Sign up with GitHub
   - Click "New site from Git"
   - Select your repository
2. **Configure Build Settings**:
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/build`
3. **Environment Variables**:
   - Add `REACT_APP_API_URL` = `https://your-railway-backend-url.railway.app`
4. **Deploy**: Click "Deploy site"

#### Backend (Railway)
- Follow the same Railway steps as Option 1

**Cost**: Free (Netlify: 100GB bandwidth/month, Railway: $5 credit/month)

---

## 🐳 Docker Deployment (Advanced)

### Deploy to Railway with Docker
1. **Push to GitHub** with the included Dockerfile
2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
3. **Configure Docker**:
   - Railway will detect the Dockerfile
   - Set Port: `5000`
4. **Deploy**: Railway will build and deploy the Docker container

### Deploy to Render with Docker
1. **Connect to Render** with your GitHub repo
2. **Select Docker** as the environment
3. **Configure**:
   - Dockerfile path: `./Dockerfile`
   - Port: `5000`
4. **Deploy**: Render will build and deploy the container

---

## 🔧 Configuration Steps

### 1. Update API URLs

After deploying your backend, update the frontend configuration:

**For Vercel/Netlify**:
```bash
# In your frontend/.env file
REACT_APP_API_URL=https://your-backend-url.railway.app
```

**For Render**:
```bash
# In your frontend/.env file
REACT_APP_API_URL=https://alpha-signal-backend.onrender.com
```

### 2. Environment Variables

Set these in your deployment platform:

**Required**:
- `FLASK_ENV=production`
- `PYTHONPATH=/app` (or your project path)

**Optional** (for live trading):
- `ALPACA_API_KEY=your_api_key`
- `ALPACA_SECRET_KEY=your_secret_key`

### 3. CORS Configuration

The backend already includes CORS configuration, but if you encounter issues:

```python
# In backend/app.py
CORS(app, origins=["https://your-frontend-domain.vercel.app"])
```

---

## 🚀 Quick Start Commands

### Local Testing
```bash
# Test the Docker setup locally
docker-compose up --build

# Test individual components
cd frontend && npm start
cd backend && python app.py
```

### Deployment Commands
```bash
# Push to GitHub (required for all platforms)
git add .
git commit -m "Deploy to production"
git push origin main

# For Railway CLI (optional)
railway login
railway link
railway up
```

---

## 📊 Performance Optimization

### Frontend Optimization
1. **Build Optimization**:
   ```bash
   cd frontend
   npm run build
   # The build folder will be optimized for production
   ```

2. **Environment Variables**:
   ```bash
   # Create frontend/.env.production
   REACT_APP_API_URL=https://your-backend-url
   GENERATE_SOURCEMAP=false
   ```

### Backend Optimization
1. **Production Settings**:
   ```python
   # In backend/app.py
   app.run(host='0.0.0.0', port=5000, debug=False)
   ```

2. **Memory Optimization**:
   - The free tiers have limited memory
   - Consider reducing data processing for large datasets
   - Use pagination for large results

---

## 🔍 Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure CORS is properly configured in backend
   - Check that API URLs are correct

2. **Build Failures**:
   - Check that all dependencies are in requirements.txt
   - Ensure Python version compatibility (3.8+)

3. **Memory Issues**:
   - Free tiers have limited RAM
   - Consider reducing dataset sizes for backtesting

4. **Sleep Mode** (Render):
   - Render free tier sleeps after 15 minutes
   - First request after sleep may be slow
   - Consider upgrading to paid plan for always-on

### Debug Commands
```bash
# Check backend health
curl https://your-backend-url.railway.app/api/health

# Check frontend
curl https://your-frontend-url.vercel.app

# Local testing
python backend/app.py
cd frontend && npm start
```

---

## 💰 Cost Comparison

| Platform | Frontend | Backend | Total Cost |
|----------|----------|---------|------------|
| Vercel + Railway | Free | Free* | $0/month |
| Render | Free | Free* | $0/month |
| Netlify + Railway | Free | Free* | $0/month |

*Free tiers have usage limits but are sufficient for development and small-scale production use.

---

## 🎉 Success!

Once deployed, your Alpha Signal Engine will be available at:
- **Frontend**: `https://your-app.vercel.app` (or your chosen platform)
- **Backend API**: `https://your-backend.railway.app` (or your chosen platform)

Your quantitative trading platform is now live on the internet! 🚀

---

## 📞 Support

If you encounter issues:
1. Check the platform-specific documentation
2. Review the logs in your deployment platform
3. Test locally first with `docker-compose up`
4. Ensure all environment variables are set correctly

Happy trading! 📈
