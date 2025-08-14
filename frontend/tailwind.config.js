/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './public/index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: '#14b8a6',
          teal: '#14b8a6',
          blue: '#38bdf8',
        },
      },
      boxShadow: {
        card: '0 10px 25px -10px rgba(0,0,0,0.45)'
      }
    },
  },
  plugins: [],
};


