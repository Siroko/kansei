import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

import { extname, relative, resolve } from 'path'
import { fileURLToPath } from 'node:url'
import { glob } from 'glob'
import mkcert from 'vite-plugin-mkcert'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        dts({ include: ['src'] }),
        mkcert()
    ],
    build: {
        rollupOptions: {
            external: [],
            input: Object.fromEntries(
                glob.sync('src/**/*.{ts,tsx}', {
                    ignore: ["src/**/*.d.ts"],
                }).map(file => [
                    // The name of the entry point
                    // src/nested/foo.ts becomes nested/foo
                    relative(
                        'src',
                        file.slice(0, file.length - extname(file).length)
                    ),
                    // The absolute path to the entry file
                    // lib/nested/foo.ts becomes /project/lib/nested/foo.ts
                    fileURLToPath(new URL(file, import.meta.url))
                ])
            ),
            output: {
                assetFileNames: 'assets/[name][extname]',
                entryFileNames: '[name].js',
            }
        },
        lib: {
            entry: resolve(__dirname, 'src/main.ts'),
            formats: ['es']
        },
        copyPublicDir: false,
    },
    resolve: { alias: { src: resolve('src/') } },
})