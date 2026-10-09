// Planted-foot slide of a motion-matching pack over the scripted course, on the TS engine
// (`src/animation/motion_matching/FootSlide.ts`): the twin of Rust's
// `cargo run -p kansei-core --release --example foot_slide -- <pack.kmm>`, same table.
// Usage: node scripts/foot-slide.mjs <pack.kmm> [hero=<character.kmm>] [walk=2] [run=5] [only=<name part>]
// Packs are private: link them in, never commit them.
import { readFileSync } from 'node:fs'
import process from 'node:process'
import { createServer } from 'vite'

const args = process.argv.slice(2)
const arg = (key) => args.find((a) => a.startsWith(`${key}=`))?.slice(key.length + 1)
const path = args.find((a) => !a.includes('='))
// a copy: `Buffer`s from a pool may start off a 4-byte boundary, and the reader views floats in place
const read = (p) => new Uint8Array(readFileSync(p))
if (!path) {
    console.error('usage: node scripts/foot-slide.mjs <pack.kmm> [hero=<character.kmm>] [walk=2] [run=5] [only=<name part>]')
    process.exit(1)
}
const server = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, ws: false }, appType: 'custom', logLevel: 'error' })
try {
    const mm = await server.ssrLoadModule('/src/animation/motion_matching/index.ts')
    const { Retarget } = await server.ssrLoadModule('/src/animation/Retarget.ts')
    const pack = mm.MotionPack.fromBytes(read(path))
    const hero = arg('hero') && mm.CharacterPack.fromBytes(read(arg('hero')))
    const db = pack.database
    const reference = mm.MotionPack.fromBytes(read(path)).database
    reference.detectContacts(mm.defaultContactThresholds())
    // the demo's gait filter: idle + walk or idle + run by the pack's tags
    const tags = (pack.metaValue('tags') ?? '').split(',')
    const bit = (name) => (tags.indexOf(name) < 0 ? 0 : 1 << tags.indexOf(name))
    const [idle, walk, run] = [bit('idle'), bit('walk'), bit('run')]
    const gait = walk !== 0 && run !== 0
    const pad = (s, n) => String(s).padEnd(n)
    const num = (x, n) => x.toFixed(1).padStart(n)
    console.log(`${pad('scenario', 26)} ${'cm/s'.padStart(7)} ${'cm/plant'.padStart(9)} ${'planted'.padStart(8)} ${'mesh off'.padStart(9)} ${'max off'.padStart(9)}`)
    const reports = []
    for (const scenario of mm.footSlideCourse(Number(arg('walk') ?? 2), Number(arg('run') ?? 5)).filter((s) => !arg('only') || s.name.includes(arg('only')))) {
        const matcher = new mm.MotionMatcher(db, mm.defaultMotionMatchingSettings())
        matcher.settings.filter.tags = gait ? idle | (scenario.run ? run : walk) : ~mm.ACTION_TAG >>> 0
        if (hero) matcher.setDisplay(db, { skeleton: hero.skeleton, retarget: new Retarget(db.skeleton, hero.skeleton, Retarget.UNREAL_KEEP) })
        const r = mm.measureFootSlide(db, reference, matcher, scenario)
        reports.push(r)
        console.log(`${pad(scenario.name, 26)} ${num(r.cmPerSecond, 7)} ${num(r.cmPerPlant, 9)} ${(100 * r.planted).toFixed(0).padStart(7)}% ${num(r.yawGap, 8)}° ${num(r.yawGapMax, 8)}°`)
    }
    const mean = (f) => reports.reduce((s, r) => s + f(r), 0) / Math.max(reports.length, 1)
    console.log(`${pad('mean', 26)} ${num(mean((r) => r.cmPerSecond), 7)} ${num(mean((r) => r.cmPerPlant), 9)}`)
} finally {
    await server.close()
}
