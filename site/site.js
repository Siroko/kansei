// Kansei Graphics landing page: lazy-load the Raggare loop and keep the footer year current.
;(() => {
  const year = document.getElementById('year')
  if (year) year.textContent = String(new Date().getFullYear())

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  // The hero loop autoplays; honour reduced motion by stopping it (CSS also hides it).
  const hero = document.querySelector('.hero-video')
  if (hero && reduceMotion) {
    hero.removeAttribute('autoplay')
    hero.pause()
  }

  // Load the Raggare clip only when it scrolls near view, and pause it when off screen.
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
})()
