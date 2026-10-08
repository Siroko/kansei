import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'
import wasm from 'vite-plugin-wasm'

import { readFileSync } from 'fs'
import { extname, relative, resolve } from 'path'
import { fileURLToPath } from 'node:url'
import { glob } from 'glob'
import mkcert from 'vite-plugin-mkcert'
import { devSite } from './scripts/vite-dev-site'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        dts({
            include: ['src'],
            // The declarations use WebGPU's global types: load them for the package's users too.
            beforeWriteFile: (filePath, content) => filePath.endsWith('/dist/main.d.ts')
                ? { content: `/// <reference types="@webgpu/types" />\n${content}` }
                : undefined,
        }),
        mkcert(),
        wasm(),
        devSite(__dirname),
        {
            // The KTX2 loader inlines the vendored Basis Universal transcoder (Apache-2.0): ship its
            // licence beside it in the package.
            name: 'basis-licence',
            apply: 'build',
            generateBundle() {
                this.emitFile({
                    type: 'asset',
                    fileName: 'loaders/ktx2/basis/LICENSE',
                    source: readFileSync(resolve(__dirname, 'src/loaders/ktx2/basis/LICENSE')),
                })
            },
        },
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
        copyPublicDir: false
    },
    resolve: { alias: { src: resolve('src/') } },
})