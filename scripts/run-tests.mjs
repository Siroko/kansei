// Run the CPU-only TypeScript tests (tests/**/*.test.ts) in Node, through Vite's module loader
// (TypeScript and the `src` imports as the examples use them). Exits non-zero on a failure.
// Usage: pnpm test [filter]   (only test files whose path contains the filter)
import process from 'node:process'
import { createServer } from 'vite'
import { glob } from 'glob'

const filter = process.argv[2] ?? ''
const files = glob.sync('tests/**/*.test.ts').sort().filter((f) => f.includes(filter))
const server = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error' })
try {
    const { runAll } = await server.ssrLoadModule('/tests/harness.ts')
    const failed = await runAll(files.map((f) => [f, () => server.ssrLoadModule('/' + f)]))
    process.exitCode = failed > 0 ? 1 : 0
} finally {
    await server.close()
}
