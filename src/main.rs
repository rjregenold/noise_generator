extern crate portaudio;

use fastrand;
use portaudio as pa;
use std::env;
use std::io;
use std::time::Instant;

const CHANNELS: i32 = 2;
const SAMPLE_RATE: f64 = 44_100.0;
const FADE_IN_SECONDS: f32 = 10.0;

/// The max random value when generating a u16, represented
/// as a f32. This helps to generate random f32s in a given
/// range.
const RAND_MAX: f32 = u16::MAX as f32;

// --
// A pink noise generator, ported from:
// https://github.com/PortAudio/portaudio/blob/master/examples/paex_pink.c

const MAX_RANDOM_ROWS: usize = 30;
const PINK_RANDOM_BITS: usize = 24;
const PINK_RANDOM_SHIFT: usize = (std::mem::size_of::<i64>() * 8) - PINK_RANDOM_BITS;

struct PinkNoise {
    rows: [i64; MAX_RANDOM_ROWS],
    running_sum: i64,
    index: i32,
    index_mask: i32,
    scalar: f32,
}

fn init_pink_noise(num_rows: usize) -> PinkNoise {
    // the maximum possible signed random value
    let pmax = (num_rows + 1) * (1 << (PINK_RANDOM_BITS - 1));

    PinkNoise {
        rows: [0i64; MAX_RANDOM_ROWS],
        running_sum: 0i64,
        index: 0,
        index_mask: (1 << num_rows) - 1,
        scalar: 1f32 / pmax as f32,
    }
}

fn generate_pink_noise(rng: &fastrand::Rng, pink_noise: &mut PinkNoise) -> f32 {
    let mut new_random: i64;

    pink_noise.index = (pink_noise.index + 1) & pink_noise.index_mask;

    if pink_noise.index != 0 {
        let mut num_zeroes = 0;
        let mut n = pink_noise.index;

        while (n & 1) == 0 {
            n = n >> 1;
            num_zeroes += 1;
        }

        pink_noise.running_sum -= pink_noise.rows[num_zeroes];

        new_random = rng.i64(..) >> PINK_RANDOM_SHIFT;
        pink_noise.running_sum += new_random;
        pink_noise.rows[num_zeroes] = new_random;
    }

    new_random = rng.i64(..) >> PINK_RANDOM_SHIFT;
    let sum = pink_noise.running_sum + new_random;
    let output = sum as f32 * pink_noise.scalar;

    // attempt to increase amplitude
    f32::min(1f32, output * 2f32)
}

// --

struct BrownNoise {
    running_sum: f32,
}

fn init_brown_noise(rng: &fastrand::Rng) -> BrownNoise {
    BrownNoise {
        running_sum: randf(rng, -1f32, 1f32),
    }
}

fn generate_brown_noise(rng: &fastrand::Rng, brown_noise: &mut BrownNoise) -> f32 {
    let (a, b) = match brown_noise.running_sum {
        sum if sum < -0.95 => (0.1, 0.2),
        sum if sum > 0.95 => (-0.2, -0.1),
        _ => (-0.1, 0.1),
    };

    let offset = randf(rng, a, b);

    brown_noise.running_sum += offset;

    f32::max(f32::min(1f32, brown_noise.running_sum), -1f32)
}

/// Represents the type of noise to generate.
#[derive(Debug)]
enum NoiseType {
    White,
    Pink,
    Brown,
}

/// The app config.
struct Config {
    noise_type: NoiseType,
}

impl Config {
    /// Generates a Config from command line arguments.
    fn new(mut args: env::Args) -> Config {
        args.next();

        let noise_type = match args.next().as_ref().map(String::as_str) {
            Some("pink") => NoiseType::Pink,
            Some("brown") => NoiseType::Brown,
            _ => NoiseType::White,
        };

        Config { noise_type }
    }
}

/// Generates a random f32 between low and high.
fn randf(rng: &fastrand::Rng, low: f32, high: f32) -> f32 {
    (rng.u16(..) as f32 / RAND_MAX) * f32::abs(low - high) + low
}

fn main() {
    let config = Config::new(env::args());

    match run(config) {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run: {:?}", e);
        }
    }
}

fn run(config: Config) -> Result<(), pa::Error> {
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

    let rng = fastrand::Rng::with_seed(777);
    let mut fade_in_scalar = 0.0;
    let mut now_m = None;

    let mut left_pink = init_pink_noise(12);
    let mut right_pink = init_pink_noise(16);

    let mut left_brown = init_brown_noise(&rng);
    let mut right_brown = init_brown_noise(&rng);

    let msg = format!(
        "playing {} noise. press <enter> to stop.",
        match config.noise_type {
            NoiseType::White => "white",
            NoiseType::Pink => "pink",
            NoiseType::Brown => "brown",
        }
    )
    .to_string();

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

        match config.noise_type {
            NoiseType::White => {
                let num_samples = frames * 2;
                for i in 0..num_samples {
                    buffer[i] = randf(&rng, -1f32, 1f32) * fade_in_scalar;
                }
            }

            NoiseType::Pink => {
                let mut i = 0;
                for _ in 0..frames {
                    buffer[i] = generate_pink_noise(&rng, &mut left_pink) * fade_in_scalar;
                    buffer[i + 1] = generate_pink_noise(&rng, &mut right_pink) * fade_in_scalar;
                    i += 2
                }
            }

            NoiseType::Brown => {
                let mut i = 0;
                for _ in 0..frames {
                    buffer[i] = generate_brown_noise(&rng, &mut left_brown) * fade_in_scalar;
                    buffer[i + 1] = generate_brown_noise(&rng, &mut right_brown) * fade_in_scalar;
                    i += 2
                }
            }
        }

        pa::Continue
    };

    let mut stream = pa.open_non_blocking_stream(settings, callback)?;

    stream.start()?;

    println!("{}", msg);

    let mut _input = String::new();
    let _ = io::stdin().read_line(&mut _input);

    stream.stop()?;
    stream.close()?;

    Ok(())
}
