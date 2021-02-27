extern crate portaudio;

use portaudio as pa;
use std::io;
use std::time::Instant;

const CHANNELS: i32 = 2;
const SAMPLE_RATE: f64 = 44_100.0;
const FADE_IN_SECONDS: f32 = 10.0;

// ---
// a linear congruential generator that generates pseudo-random numbers
// very quickly.
//
// https://en.wikipedia.org/wiki/Linear_congruential_generator
//
// the parameters are chosen based on the Numerical Recipes book.

struct LGC {
    modulus: u64,
    a: u64,
    c: u64,
    state: u64,
}

impl LGC {
    fn new(seed: u64) -> LGC {
        LGC {
            modulus: 4294967296,
            a: 1664525,
            c: 1013904223,
            state: seed,
        }
    }
}

impl Iterator for LGC {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.state = (self.state * self.a + self.c) % self.modulus;

        return Some(self.state);
    }
}

// ---

fn main() {
    match run() {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run: {:?}", e);
        }
    }
}

fn next_sample(rng: &mut LGC) -> f32 {
    // scales the next pseudo-random number to a value between -1 and 1
    return rng
        .next()
        .map(|x| -1.0 + 2f32 * (x as f32) / (rng.modulus as f32))
        .unwrap_or(0f32);
}

fn run() -> Result<(), pa::Error> {
    let pa = pa::PortAudio::new()?;

    let device = pa.default_output_device()?;
    let output_info = pa.device_info(device)?;
    println!("Default output device info: {:#?}", &output_info);

    // the default latency on the raspberry pi is not correct.
    // https://github.com/PortAudio/portaudio/issues/246
    // let latency = output_info.default_high_output_latency;

    // 100ms latency
    let latency = 0.1;

    let params = pa::StreamParameters::new(device, CHANNELS, true, latency);
    let mut settings =
        pa::OutputStreamSettings::new(params, SAMPLE_RATE, pa::FRAMES_PER_BUFFER_UNSPECIFIED);
    settings.flags = pa::stream_flags::CLIP_OFF;

    let mut rng = LGC::new(777);
    let mut fade_in_scalar = 0.0;
    let mut now_m = None;

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        // there is a bug in the portaudio alsa api that makes
        // the `time` argument empty, so we use the system clock
        // https://github.com/PortAudio/portaudio/issues/498
        let now = now_m.get_or_insert(Instant::now());
        let elapsed = now.elapsed();

        if fade_in_scalar < 1.0 {
            let delta = elapsed.as_secs_f32();

            fade_in_scalar = ((delta / FADE_IN_SECONDS) + 1.0).log2().min(1.0);
        }

        let num_samples = frames * 2;
        for i in 0..num_samples {
            buffer[i] = next_sample(&mut rng) * fade_in_scalar;
        }

        pa::Continue
    };

    let mut stream = pa.open_non_blocking_stream(settings, callback)?;

    stream.start()?;

    println!("playing white noise. press <enter> to stop.");
    let mut _input = String::new();
    let _ = io::stdin().read_line(&mut _input);

    stream.stop()?;
    stream.close()?;

    Ok(())
}
