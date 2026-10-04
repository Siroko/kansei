# Joy Division

Unknown Pleasures' pulsar lines driven by the spectrum of *A Means to an End*, with its lyrics.

The recording is not in this repository and the example is not published on kansei.graphics
(`scripts/build-wasm-examples.sh` skips it). Bring your own copy of the track:

```sh
wasm-pack build --target web --release --out-dir www/pkg
cd www && python3 -m http.server 8080
```

Open http://localhost:8080/ and either

- drop the audio file onto the page, or click *choose an audio file*; or
- pass a URL: `http://localhost:8080/?audio=assets/audio/a-means-to-an-end.ogg`. A file under
  `www/assets/audio/` is git-ignored, so a local copy there stays out of commits. A URL on another
  origin (a private bucket, say) must allow this page through CORS.

Click the canvas to play or pause. The lyric timings in `lyrics.json` follow the album version.
Any format the browser decodes works (mp3, ogg, wav, m4a, flac).
