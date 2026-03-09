import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // Ensure all routes are handled by index.html for client-side routing
  build: {
    rollupOptions: {
      input: {
        main: './index.html',
      },
    },
  },
})
