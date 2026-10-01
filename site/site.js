// Kansei Graphics landing page: the live lake behind the hero, the lazy Raggare clip and the footer year.
;(() => {
  const year = document.getElementById('year')
  if (year) year.textContent = String(new Date().getFullYear())

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  liveBackground()
  lazyVideos()

  /*
   * Hero background. The hero starts black and one background fades in from black:
   *   1. the live lake from the Raggare web teaser, in an iframe, once it says it is ready;
   *   2. otherwise (small or touch screens, no WebGPU, ?bg=off, or no ready message within 8 s) a muted loop cut from
   *      the teaser's film, once it can play through;
   *   3. with reduced motion, or if the loop cannot play, a still of the lake.
   * Messages, all { type, ... }:
   *   iframe -> page  { type: 'kansei-bg-ready' }
   *   page -> iframe  { type: 'kansei-bg-pointer', x, y }  pointer over the page, -1..1: x left to right, y top to bottom
   *   page -> iframe  { type: 'kansei-bg-pause' } / { type: 'kansei-bg-resume' }  tab or hero hidden / shown again
   * ?bg=<url> swaps the iframe for local testing (localhost, 127.0.0.1 or *.kansei.graphics only); ?bg=off skips it.
   */
  function liveBackground() {
    const media = document.querySelector('.hero-media')
    const still = document.querySelector('.hero-still')
    const video = document.querySelector('.hero-video')
    if (!media) return

    const DEFAULT_SRC = 'https://raggare.kansei.graphics/bg/lake/'
    const READY_TIMEOUT_MS = 8000
    const VIDEO_TIMEOUT_MS = 15000
    const DRIFT_PX = 14

    const small = window.matchMedia('(max-width: 760px), (pointer: coarse)').matches

    // Fallbacks fetch nothing until they are needed.
    const showStill = () => {
      if (!still || still.classList.contains('is-shown')) return
      const source = still.parentElement.querySelector('source[data-srcset]')
      if (source) source.srcset = source.dataset.srcset
      still.src = still.dataset.src
      const reveal = () => requestAnimationFrame(() => still.classList.add('is-shown'))
      if (still.complete && still.naturalWidth) reveal()
      else still.addEventListener('load', reveal, { once: true })
    }

    let videoOn = false
    const showVideo = () => {
      if (!video) return showStill()
      if (videoOn) return
      videoOn = true
      const sources = small
        ? [['/media/lake-loop-960.mp4', 'video/mp4']]
        : [
            ['/media/lake-loop.webm', 'video/webm'],
            ['/media/lake-loop.mp4', 'video/mp4'],
          ]
      for (const [url, type] of sources) {
        const el = document.createElement('source')
        el.src = url
        el.type = type
        video.appendChild(el)
      }
      let slow = 0
      const fail = () => {
        clearTimeout(slow)
        if (video.classList.contains('is-shown')) return
        video.remove()
        showStill()
      }
      const reveal = () => {
        if (video.classList.contains('is-shown')) return
        clearTimeout(slow)
        requestAnimationFrame(() => video.classList.add('is-shown'))
        syncPause()
      }
      // Fade in once it can play through (or, where that event waits for playback, once playback is under way).
      video.addEventListener('canplaythrough', reveal, { once: true })
      video.addEventListener('timeupdate', function onTime() {
        if (video.currentTime < 0.3) return
        video.removeEventListener('timeupdate', onTime)
        reveal()
      })
      // Low-power modes can refuse to autoplay; if it never gets going, show the still instead.
      slow = setTimeout(fail, VIDEO_TIMEOUT_MS)
      video.querySelector('source:last-child').addEventListener('error', fail, { once: true })
      video.preload = 'auto'
      video.load()
      video.play().catch((err) => {
        if (err && err.name === 'NotAllowedError') fail()
      })
    }

    if (reduceMotion) {
      showStill()
      return
    }

    const src = backgroundSrc(DEFAULT_SRC)
    const canRunLive = src && !small && 'gpu' in navigator

    let frame = null
    let live = false
    let pending = null
    let raf = 0

    const post = (msg) => {
      if (live && frame && frame.contentWindow) frame.contentWindow.postMessage(msg, src.origin)
    }

    // The pointer steers the live camera, or drifts the loop or still a little. One update per frame.
    if (window.matchMedia('(pointer: fine)').matches) {
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
      } else {
        media.style.setProperty('--px', `${(-x * DRIFT_PX).toFixed(1)}px`)
        media.style.setProperty('--py', `${(-y * DRIFT_PX * 0.6).toFixed(1)}px`)
      }
    }

    // Pause the scene or the loop while the tab is hidden or the hero is scrolled away.
    let heroVisible = true
    let paused = false
    const syncPause = () => {
      const shouldPause = document.hidden || !heroVisible
      if (videoOn && video && video.isConnected) {
        if (shouldPause) video.pause()
        else if (video.classList.contains('is-shown')) video.play().catch(() => {})
      }
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

    if (!canRunLive) {
      showVideo()
      return
    }

    frame = document.createElement('iframe')
    frame.className = 'hero-live'
    frame.title = 'Live background'
    frame.setAttribute('aria-hidden', 'true')
    frame.setAttribute('tabindex', '-1')
    frame.setAttribute('referrerpolicy', 'strict-origin')
    frame.src = src.href

    const giveUp = setTimeout(() => {
      if (live) return
      window.removeEventListener('message', onMessage)
      frame.remove()
      frame = null
      showVideo()
    }, READY_TIMEOUT_MS)

    function onMessage(e) {
      if (!frame || e.source !== frame.contentWindow || e.origin !== src.origin) return
      if (!e.data || e.data.type !== 'kansei-bg-ready' || live) return
      clearTimeout(giveUp)
      live = true
      frame.classList.add('is-ready')
      syncPause()
    }

    // The hero is black until something is ready, so start the scene straight away (this script is deferred).
    window.addEventListener('message', onMessage)
    media.appendChild(frame)
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
      video.preload = 'auto'
      video.load()
      video.play().catch((err) => {
        if (err && err.name === 'NotAllowedError') fail()
      })
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
