// Kansei Graphics landing page: the live lake behind the hero, the lazy Raggare clip and the footer year.
;(() => {
  const year = document.getElementById('year')
  if (year) year.textContent = String(new Date().getFullYear())

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  liveBackground()
  lazyVideos()

  /*
   * Hero background. A static still of the lake is always there. Where it can run, the live lake from the Raggare
   * demo loads in an iframe on top and fades in once it says it is ready. Messages, all { type, ... } objects:
   *   iframe -> page  { type: 'kansei-bg-ready' }
   *   page -> iframe  { type: 'kansei-bg-pointer', x, y }  pointer over the page, -1..1: x left to right, y top to bottom
   *   page -> iframe  { type: 'kansei-bg-pause' } / { type: 'kansei-bg-resume' }  tab or hero hidden / shown again
   * ?bg=<url> swaps the iframe for local testing (localhost, 127.0.0.1 or *.kansei.graphics only); ?bg=off skips it.
   */
  function liveBackground() {
    const media = document.querySelector('.hero-media')
    const still = document.querySelector('.hero-still')
    if (!media || reduceMotion) return

    const DEFAULT_SRC = 'https://raggare.kansei.graphics/bg/lake/'
    const READY_TIMEOUT_MS = 8000
    const DRIFT_PX = 14

    const small = window.matchMedia('(max-width: 760px), (pointer: coarse)').matches
    const src = backgroundSrc(DEFAULT_SRC)
    const canRunLive = src && !small && 'gpu' in navigator

    let frame = null
    let live = false
    let pending = null
    let raf = 0

    const post = (msg) => {
      if (live && frame && frame.contentWindow) frame.contentWindow.postMessage(msg, src.origin)
    }

    // The pointer drifts the still a little, and steers the live camera once it is up. One update per frame.
    const fine = window.matchMedia('(pointer: fine)').matches
    if (fine || canRunLive) {
      window.addEventListener(
        'pointermove',
        (e) => {
          pending = {
            x: clamp((e.clientX / window.innerWidth) * 2 - 1),
            y: clamp((e.clientY / window.innerHeight) * 2 - 1),
          }
          if (!raf) raf = requestAnimationFrame(flush)
        },
        { passive: true }
      )
    }

    function flush() {
      raf = 0
      if (!pending) return
      const { x, y } = pending
      pending = null
      if (live) {
        post({ type: 'kansei-bg-pointer', x, y })
      } else if (still) {
        still.style.setProperty('--px', `${(-x * DRIFT_PX).toFixed(1)}px`)
        still.style.setProperty('--py', `${(-y * DRIFT_PX * 0.6).toFixed(1)}px`)
      }
    }

    if (!canRunLive) return

    const start = () => {
      frame = document.createElement('iframe')
      frame.className = 'hero-live'
      frame.title = 'Live background'
      frame.setAttribute('aria-hidden', 'true')
      frame.setAttribute('tabindex', '-1')
      frame.setAttribute('loading', 'eager')
      frame.setAttribute('referrerpolicy', 'strict-origin')
      frame.src = src.href

      const giveUp = setTimeout(() => {
        if (live) return
        window.removeEventListener('message', onMessage)
        frame.remove()
        frame = null
      }, READY_TIMEOUT_MS)

      function onMessage(e) {
        if (!frame || e.source !== frame.contentWindow || e.origin !== src.origin) return
        if (!e.data || e.data.type !== 'kansei-bg-ready' || live) return
        clearTimeout(giveUp)
        live = true
        frame.classList.add('is-ready')
        syncPause()
      }

      window.addEventListener('message', onMessage)
      media.appendChild(frame)
    }

    // Pause the scene while the tab is hidden or the hero is scrolled away.
    let heroVisible = true
    let paused = false
    const syncPause = () => {
      const shouldPause = document.hidden || !heroVisible
      if (!live || shouldPause === paused) return
      paused = shouldPause
      post({ type: paused ? 'kansei-bg-pause' : 'kansei-bg-resume' })
    }
    document.addEventListener('visibilitychange', syncPause)
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([entry]) => {
        heroVisible = entry.isIntersecting
        syncPause()
      }).observe(media)
    }

    // Start after the page itself has loaded so the scene never competes with the first paint.
    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 300))
    if (document.readyState === 'complete') idle(start)
    else window.addEventListener('load', () => idle(start), { once: true })
  }

  function backgroundSrc(fallback) {
    let wanted = fallback
    try {
      const param = new URLSearchParams(window.location.search).get('bg')
      if (param === 'off') return null
      if (param) wanted = param
      const url = new URL(wanted, window.location.href)
      const host = url.hostname
      const allowed =
        (url.protocol === 'https:' || url.protocol === 'http:') &&
        (host === 'localhost' || host === '127.0.0.1' || host === 'kansei.graphics' || host.endsWith('.kansei.graphics'))
      return allowed ? url : new URL(fallback)
    } catch {
      return new URL(fallback)
    }
  }

  function clamp(v) {
    return Math.max(-1, Math.min(1, Math.round(v * 1000) / 1000))
  }

  // Load the Raggare clip only when it scrolls near view, and pause it when off screen.
  function lazyVideos() {
    const videos = document.querySelectorAll('video.lazy-video')
    if (reduceMotion || !('IntersectionObserver' in window)) return

    const load = (video) => {
      if (video.dataset.loaded) return
      video.querySelectorAll('source[data-src]').forEach((s) => {
        s.src = s.dataset.src
      })
      video.dataset.loaded = '1'
      video.load()
    }

    const io = new IntersectionObserver(
      (entries) => {
        for (const { target, isIntersecting } of entries) {
          if (isIntersecting) {
            load(target)
            target.play().catch(() => {})
          } else if (target.dataset.loaded) {
            target.pause()
          }
        }
      },
      { rootMargin: '200px 0px' }
    )
    videos.forEach((v) => io.observe(v))
  }
})()
