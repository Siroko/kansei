// Dev-server stand-in for what the site build (`pnpm bundle-examples`) lays out in dist/: site/
// at the root (so /site.css, /favicon.svg and the landing page resolve, as on kansei.graphics) and
// the prebuilt Rust/WASM examples at /examples/<name>/. Those come from a local
// `scripts/build-wasm-examples.sh` output, `WASM_EXAMPLES_DIR` or build/wasm-examples; without
// one they 404 and examples/index.html points their links at kansei.graphics instead.
import type { Plugin } from 'vite'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, resolve, sep } from 'node:path'

const TYPES: Record<string, string> = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.webp': 'image/webp',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
    '.webm': 'video/webm',
    '.mp4': 'video/mp4',
}

// The file `urlPath` names under `dir` (a directory's index.html), or undefined.
function fileIn(dir: string, urlPath: string): string | undefined {
    const file = resolve(join(dir, decodeURIComponent(urlPath)))
    if (file !== dir && !file.startsWith(dir + sep)) return undefined
    if (!existsSync(file)) return undefined
    const stat = statSync(file)
    if (stat.isFile()) return file
    const index = join(file, 'index.html')
    return stat.isDirectory() && existsSync(index) ? index : undefined
}

export function devSite(root: string): Plugin {
    const site = resolve(root, 'site')
    const wasm = resolve(root, process.env.WASM_EXAMPLES_DIR ?? 'build/wasm-examples')
    return {
        name: 'kansei-dev-site',
        apply: 'serve',
        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                if (req.method !== 'GET' && req.method !== 'HEAD') return next()
                const path = (req.url ?? '/').split('?')[0]
                const rust = path.match(/^\/examples\/([^/]+)(\/.*)?$/)
                let file: string | undefined
                if (rust && !rust[1].includes('.')) {
                    // A Rust example's folder without its trailing slash would break its relative URLs.
                    if (!rust[2] && fileIn(wasm, rust[1])) {
                        res.writeHead(301, { Location: `/examples/${rust[1]}/` })
                        return res.end()
                    }
                    file = fileIn(wasm, rust[1] + (rust[2] ?? ''))
                } else if (!path.startsWith('/examples/') && !path.startsWith('/src/')) {
                    file = fileIn(site, path)
                }
                if (!file) return next()
                res.writeHead(200, {
                    'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
                    'Cache-Control': 'no-store',
                })
                if (req.method === 'HEAD') return res.end()
                createReadStream(file).pipe(res)
            })
        },
    }
}
