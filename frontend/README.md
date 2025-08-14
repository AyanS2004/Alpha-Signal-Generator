# Alpha Signal Engine Frontend

A modern React-based web interface for the Alpha Signal Engine, providing an intuitive and powerful trading signal generation and backtesting platform.

## 🚀 Features

### 📊 Dashboard
- Real-time performance metrics
- Interactive equity curve visualization
- Current trading signal display
- Key performance indicators (Sharpe ratio, drawdown, win rate)

### 🔬 Backtesting
- Drag-and-drop CSV file upload
- Comprehensive backtest results
- Performance metrics visualization
- Signal analysis and trade history

### ⚡ Real-time Trading
- Live market data feeds
- Real-time signal generation
- Auto-trading capabilities
- Signal confidence tracking

### 🔧 Parameter Optimization
- Interactive parameter sliders
- Multi-parameter optimization
- Performance comparison charts
- Best parameter application

### ⚙️ Settings
- Trading configuration
- Signal parameters
- System settings
- API credentials management

## 🛠️ Technology Stack

- **Frontend**: React 18, TypeScript
- **UI Framework**: Material-UI (MUI) v5
- **Charts**: Recharts
- **Animations**: Framer Motion
- **File Upload**: React Dropzone
- **Notifications**: React Hot Toast
- **Routing**: React Router v6

## 📦 Installation

### Prerequisites
- Node.js 16+ 
- npm or yarn
- Python 3.8+ (for backend)

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

4. **Open in browser**:
   ```
   http://localhost:3000
   ```

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start backend server**:
   ```bash
   python app.py
   ```

5. **Backend will be available at**:
   ```
   http://localhost:5000
   ```

## 🏗️ Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/
│   │   └── Layout.tsx          # Main layout with navigation
│   ├── pages/
│   │   ├── Dashboard.tsx        # Dashboard with metrics
│   │   ├── Backtesting.tsx     # Backtesting interface
│   │   ├── RealTime.tsx        # Real-time trading
│   │   ├── Optimization.tsx    # Parameter optimization
│   │   └── Settings.tsx        # Configuration settings
│   ├── App.tsx                 # Main app component
│   └── index.tsx               # Entry point
├── package.json
└── README.md
```

## 🎨 UI/UX Features

### Dark Theme
- Professional dark theme optimized for trading
- High contrast for better readability
- Consistent color scheme throughout

### Responsive Design
- Mobile-first approach
- Responsive grid layouts
- Touch-friendly controls

### Animations
- Smooth page transitions
- Loading animations
- Interactive feedback

### Real-time Updates
- Live data feeds
- Real-time signal updates
- Toast notifications

## 🔌 API Integration

The frontend communicates with the backend through REST API endpoints:

- `GET /api/health` - Health check
- `GET /api/dashboard` - Dashboard data
- `POST /api/backtest` - Run backtest
- `POST /api/optimization` - Parameter optimization
- `POST /api/realtime/start` - Start real-time feed
- `POST /api/realtime/stop` - Stop real-time feed
- `GET /api/settings` - Get settings
- `POST /api/settings` - Save settings

## 🚀 Development

### Available Scripts

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Eject from Create React App
npm run eject
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENVIRONMENT=development
```

## 🎯 Key Features

### File Upload
- Drag-and-drop CSV file upload
- File validation and error handling
- Progress indicators

### Real-time Data
- WebSocket connections for live data
- Real-time signal generation
- Auto-trading capabilities

### Parameter Optimization
- Interactive sliders for parameter adjustment
- Multi-parameter optimization
- Performance comparison charts

### Settings Management
- Comprehensive configuration options
- Secure API credential storage
- Settings persistence

## 🔒 Security

- API credentials are encrypted
- CORS protection enabled
- Input validation on all forms
- Secure file upload handling

## 📱 Mobile Support

- Responsive design for all screen sizes
- Touch-friendly interface
- Mobile-optimized navigation

## 🎨 Customization

### Theme Customization
The app uses Material-UI theming. Customize colors and styles in `src/index.tsx`:

```typescript
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    // Add your custom colors
  },
});
```

### Component Styling
All components use Material-UI's `sx` prop for styling:

```typescript
<Box sx={{ 
  display: 'flex', 
  justifyContent: 'center',
  backgroundColor: 'background.paper'
}}>
```

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure backend is running on port 5000
   - Check CORS settings
   - Verify API endpoints

2. **File Upload Issues**
   - Ensure CSV format is correct
   - Check file size limits
   - Verify required columns

3. **Real-time Feed Issues**
   - Check WebSocket connection
   - Verify API credentials
   - Ensure data provider is configured

### Debug Mode

Enable debug mode by setting:

```env
REACT_APP_DEBUG=true
```

## 📈 Performance

- Lazy loading for better performance
- Optimized bundle size
- Efficient re-rendering with React hooks
- Memoized components where appropriate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the troubleshooting section

---

**Note**: This frontend is designed to work with the Alpha Signal Engine backend. Ensure the backend is properly configured and running before using the frontend features.







