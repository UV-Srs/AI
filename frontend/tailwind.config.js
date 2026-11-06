/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          950: '#060a12',
          900: '#0b111e',
          850: '#111827',
          800: '#172033',
          700: '#23304a',
          600: '#334466',
        },
        brand: {
          teal: '#0d9488',
          cyan: '#06b6d4',
          emerald: '#10b981',
          amber: '#f59e0b',
          orange: '#f97316',
          blue: '#3b82f6',
          purple: '#8b5cf6',
        }
      }
    },
  },
  plugins: [],
}

