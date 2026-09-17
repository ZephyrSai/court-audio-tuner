# Court Audio Tuner

Browser workbench for tuning the noise suppression used by the F4E live-stream Android app.
Loops a court recording, A/Bs the processed audio against the original, and exports a config
JSON that maps 1:1 onto the app's `SpectralNoiseSuppressor`.

Everything runs client-side (Web Audio API); nothing is uploaded anywhere.

## Use

1. **Play loop** (space). **Hearing** toggles processed / original at the same playhead (`A`).
2. Pick an **algorithm** — start with the multiband expander. Raise **Floor** until the hum is gone;
   back off if anything sounds underwater or ball hits lose their snap.
3. Per-band strips add extra suppression or a static EQ trim per frequency region.
4. Narrow the **Loop** to a few seconds around a hit + voice to judge attacks.
5. **Copy config** → paste into the Android project. **Save processed audio** → zip with the 16-bit WAV
   and the config that produced it.

Drop any WAV/M4A onto the "Sample" panel to tune on a different recording.

## Algorithms

| Algorithm | Behaviour |
|---|---|
| Multiband expander | 32 log-spaced bands; each sits at a constant floor gain while only noise is present and opens when its SNR passes a threshold. No flutter by construction. |
| Band-wise Wiener | Decision-directed Wiener gain per band with rise/fall limits and time smoothing. |
| Static EQ | Low-cut + per-band static gains only. Artifact-free baseline. |
| Per-bin Wiener | The first implementation, kept so the "musical noise" crackle can be heard for comparison. |

Measurements shown: noise-floor change, hum-band change, ball-hit level change, and "blips" —
isolated one-frame gain jumps in 2–12 kHz, i.e. musical noise (original = 0).

## Build

`index.html` is self-contained (sample embedded as base64). After editing anything in `src/`:

```sh
./build.py                      # embeds samples/court-original.wav
./build.py path/to/other.wav    # embed a different default sample
```

## Layout

- `src/head.html` — markup + CSS
- `src/app.js` — DSP engine (FFT, bands, expander, Wiener, HPF), metrics, Web Audio playback, UI
- `samples/court-original.wav` — 30 s reference recording (44.1 kHz mono)
- `index.html` — built page, served by GitHub Pages
