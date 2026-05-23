import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      // КРИТИЧНО: НЕ следим за серверной частью.
      // Сервер пишет логи в training/logs/* после каждого хода — если Vite их видит,
      // он перезагружает страницу прямо во время игры. То же для адаптеров и датасетов.
      watch: {
        ignored: [
          '**/training/**',
          '**/dnd_core_models/**',
          '**/logs/**',
          '**/*.log',
          '**/*.jsonl',
          '**/__pycache__/**',
          '**/.venv/**',
          '**/trash/**',
          '**/node_modules/**',
        ],
      },
    },
  };
});
