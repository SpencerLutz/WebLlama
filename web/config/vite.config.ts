import { defineConfig } from 'vite';
import { fileURLToPath } from 'url';
import path from 'path';

// ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
    // Point to the correct location of index.html
    root: path.resolve(__dirname, '../../web'),
    
    // CSS Modules configuration
    css: {
        modules: {
            // Generate scoped class names with a hash
            generateScopedName: '[name]__[local]___[hash:base64:5]',
        },
    },
    
    server: {
        port: 3000,
        open: true,
    },
    
    build: {
        target: 'esnext',
        outDir: path.resolve(__dirname, '../../dist'),
        emptyOutDir: true,
    },
    
    resolve: {
        alias: {
            '@': path.resolve(__dirname, '../../web/src'),
            '@lib': path.resolve(__dirname, '../../src'),
        },
    },
    
    optimizeDeps: {
        exclude: ['@webgpu/types'],
    },
});