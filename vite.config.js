import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': new URL('./src', import.meta.url).pathname,
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/auth': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/aura': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})

