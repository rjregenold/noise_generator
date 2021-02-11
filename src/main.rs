extern crate portaudio;

use std::io;
use rand::distributions::{Distribution, Uniform};
use portaudio as pa;

const CHANNELS: i32 = 2;
const SAMPLE_RATE: f64 = 44_100.0;
const FRAMES_PER_BUFFER: u32 = 64;

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

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, ..}| {
        let mut idx = 0;
        for _ in 0..frames {
            buffer[idx] = between.sample(&mut rng);
            buffer[idx + 1] = between.sample(&mut rng);
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
