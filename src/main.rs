extern crate portaudio;

use std::io;
use std::time::{SystemTime};
use rand::distributions::{Distribution, Uniform};
use portaudio as pa;

const CHANNELS: i32 = 2;
const SAMPLE_RATE: f64 = 44_100.0;
const FRAMES_PER_BUFFER: u32 = 64;
const FADE_IN_SECONDS: f32 = 5.0;

fn main() {
    match run() {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run: {:?}", e);
        }
    }
}

fn run() -> Result<(), pa::Error> {
    let pa = pa::PortAudio::new()?;

    let mut settings =
        pa.default_output_stream_settings(CHANNELS, SAMPLE_RATE, FRAMES_PER_BUFFER)?;
    settings.flags = pa::stream_flags::CLIP_OFF;

    let between = Uniform::from(-1.0..1.0);
    let mut rng = rand::thread_rng();
    let mut start_time = None;
    let mut fade_in_scalar = 0.0;

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        if fade_in_scalar < 1.0 {
            let current_time = SystemTime::now();
            let started_at = start_time.get_or_insert(current_time);
            let delta = match current_time.duration_since(*started_at) {
                Ok(d) => { d.as_secs_f32() }
                _ => { FADE_IN_SECONDS }
            };

            fade_in_scalar = ((delta / FADE_IN_SECONDS) + 1.0).log2().min(1.0);
        }

        let mut idx = 0;
        for _ in 0..frames {
            buffer[idx] = between.sample(&mut rng) * fade_in_scalar;
            buffer[idx + 1] = between.sample(&mut rng) * fade_in_scalar;
            idx += 2;
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
