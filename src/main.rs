extern crate portaudio;
extern crate clap;

mod gpio;

use clap::{Arg, App};
use either::*;
use fastrand;
use portaudio as pa;
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const CHANNELS: i32 = 2;
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
#[derive(Copy, Clone, Debug)]
enum NoiseType {
    White,
    Pink,
    Brown,
}

impl NoiseType {
    fn succ(noise_type: NoiseType) -> NoiseType {
        match noise_type {
            NoiseType::White => NoiseType::Pink,
            NoiseType::Pink => NoiseType::Brown,
            NoiseType::Brown => NoiseType::White,
        }
    }
}

/// The app config.
#[derive(Clone, Debug)]
struct Config {
    enable_gpio: bool,
    settings_path: Option<String>,
}

impl Config {
    fn new() -> Config {
        let matches = App::new("Noise Generator")
                        .version("1.0")
                        .author("RJ Regenold")
                        .about("Generates different types of noise")
                        .arg(Arg::with_name("enable-gpio")
                            .short("g")
                            .long("enable-gpio")
                            .help("Enables GPIO pins"))
                        .arg(Arg::with_name("settings")
                            .short("s")
                            .long("settings")
                            .help("Path to file where app settings can be stored")
                            .takes_value(true))
                        .get_matches();

        let enable_gpio = matches.is_present("enable-gpio");
        let settings_path = matches.value_of("settings");

        Config { enable_gpio, settings_path: settings_path.map(String::from) }
    }
}

/// Generates a random f32 between low and high.
fn randf(rng: &fastrand::Rng, low: f32, high: f32) -> f32 {
    (rng.u16(..) as f32 / RAND_MAX) * f32::abs(low - high) + low
}

#[derive(Clone, Copy, Debug)]
enum VolumeLevel {
    Mute,
    _1,
    _2,
    _3,
    _4,
    _5,
    _6,
    _7,
    _8,
    _9,
    Max,
}

impl VolumeLevel {
    pub fn pred(vol: VolumeLevel) -> VolumeLevel {
        match vol {
            VolumeLevel::Mute => VolumeLevel::Mute,
            VolumeLevel::_1 => VolumeLevel::Mute,
            VolumeLevel::_2 => VolumeLevel::_1,
            VolumeLevel::_3 => VolumeLevel::_2,
            VolumeLevel::_4 => VolumeLevel::_3,
            VolumeLevel::_5 => VolumeLevel::_4,
            VolumeLevel::_6 => VolumeLevel::_5,
            VolumeLevel::_7 => VolumeLevel::_6,
            VolumeLevel::_8 => VolumeLevel::_7,
            VolumeLevel::_9 => VolumeLevel::_8,
            VolumeLevel::Max => VolumeLevel::_9,
        }
    }

    pub fn succ(vol: VolumeLevel) -> VolumeLevel {
        match vol {
            VolumeLevel::Mute => VolumeLevel::_1,
            VolumeLevel::_1 => VolumeLevel::_2,
            VolumeLevel::_2 => VolumeLevel::_3,
            VolumeLevel::_3 => VolumeLevel::_4,
            VolumeLevel::_4 => VolumeLevel::_5,
            VolumeLevel::_5 => VolumeLevel::_6,
            VolumeLevel::_6 => VolumeLevel::_7,
            VolumeLevel::_7 => VolumeLevel::_8,
            VolumeLevel::_8 => VolumeLevel::_9,
            VolumeLevel::_9 => VolumeLevel::Max,
            VolumeLevel::Max => VolumeLevel::Max,
        }
    }

    pub fn to_amp_scalar(vol: VolumeLevel) -> f32 {
        match vol {
            VolumeLevel::Mute => 0f32,
            VolumeLevel::_1 => 0.1,
            VolumeLevel::_2 => 0.2,
            VolumeLevel::_3 => 0.3,
            VolumeLevel::_4 => 0.4,
            VolumeLevel::_5 => 0.5,
            VolumeLevel::_6 => 0.6,
            VolumeLevel::_7 => 0.7,
            VolumeLevel::_8 => 0.8,
            VolumeLevel::_9 => 0.9,
            VolumeLevel::Max => 1f32,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Message {
    SetNoiseType(NoiseType),
    NextNoiseType,
    SetVolume(VolumeLevel),
    DecVolume,
    IncVolume,
}

fn main() {
    match run(Config::new()) {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run: {:?}", e);
        }
    }
}

const PIN_POWER: u8 = 5;
const PIN_NOISE_TYPE: u8 = 22;
const PIN_INC_VOLUME: u8 = 23;
const PIN_DEC_VOLUME: u8 = 24;

fn setup_gpio(tx: mpsc::Sender<Message>) -> impl FnOnce() {
    /*
    let (rx, clear_push_buttons) = gpio::setup_push_buttons_interrupt(vec![
        PIN_NOISE_TYPE,
        PIN_INC_VOLUME,
        PIN_DEC_VOLUME,
    ]);
    */

    let (rx, clear_push_buttons) = gpio::setup_push_buttons_poll(vec![
        (PIN_POWER, gpio::PushButtonBehavior::PressHold(Duration::from_secs(2), None)),
        (PIN_NOISE_TYPE, gpio::PushButtonBehavior::Switch),
        (PIN_INC_VOLUME, gpio::PushButtonBehavior::PressHold(Duration::from_secs(1), Some(Duration::from_millis(250)))),
        (PIN_DEC_VOLUME, gpio::PushButtonBehavior::PressHold(Duration::from_secs(1), Some(Duration::from_millis(250)))),
    ]);

    let cleanup = move || {
        clear_push_buttons();
    };

    std::thread::spawn(move || {
        let mut counter = 0;
        loop {
            for msg in &rx {

                match msg {
                    gpio::Message::ButtonPushed(pin) => {
                        counter = counter + 1;
                        println!("button pressed {}: {}", pin, counter);
                        match pin {
                            PIN_POWER => {
                                println!("would stop stream");
                            },
                            PIN_NOISE_TYPE => tx.send(Message::NextNoiseType).unwrap(),
                            PIN_INC_VOLUME => tx.send(Message::IncVolume).unwrap(),
                            PIN_DEC_VOLUME => tx.send(Message::DecVolume).unwrap(),
                            _ => {},
                        };
                    },
                    gpio::Message::ButtonHeld(pin) => {
                        counter = counter + 1;
                        println!("button held {}: {}", pin, counter);
                        match pin {
                            PIN_POWER => {
                                println!("would shutdown raspberry pi");
                            },
                            PIN_INC_VOLUME => tx.send(Message::IncVolume).unwrap(),
                            PIN_NOISE_TYPE => tx.send(Message::DecVolume).unwrap(),
                            _ => {},
                        }
                    },
                    gpio::Message::ButtonReleased(pin) => {
                        println!("button released {}", pin);
                    }
                    gpio::Message::Terminate => break,
                };
            }
        }
    });

    cleanup
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
        pa::OutputStreamSettings::new(params, output_info.default_sample_rate, pa::FRAMES_PER_BUFFER_UNSPECIFIED);
    settings.flags = pa::stream_flags::CLIP_OFF;

    let rng = fastrand::Rng::with_seed(777);
    let mut fade_in_scalar = 0.0;
    let mut now_m = None;

    let mut left_pink = init_pink_noise(12);
    let mut right_pink = init_pink_noise(16);

    let mut left_brown = init_brown_noise(&rng);
    let mut right_brown = init_brown_noise(&rng);

    let (tx, rx) = mpsc::channel();

    // TODO: get these from settings file.
    let mut noise_type: NoiseType = NoiseType::White;
    let mut volume_level: VolumeLevel = VolumeLevel::Max;

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        if fade_in_scalar < 1.0 {
            // there is a bug in the portaudio alsa api that makes
            // the `time` argument empty, so we use the system clock
            // https://github.com/PortAudio/portaudio/issues/498
            let now = now_m.get_or_insert(Instant::now());
            let elapsed = now.elapsed();
            let delta = elapsed.as_secs_f32();

            fade_in_scalar = ((delta / FADE_IN_SECONDS) + 1.0).log2().min(1.0);
        }

        for m in rx.try_iter() {
            match m {
                Message::SetNoiseType(nt) => noise_type = nt,
                Message::NextNoiseType => noise_type = NoiseType::succ(noise_type),
                Message::SetVolume(v) => volume_level = v,
                Message::IncVolume => volume_level = VolumeLevel::succ(volume_level),
                Message::DecVolume => volume_level = VolumeLevel::pred(volume_level),
            }
        }

        let final_scalar = fade_in_scalar * VolumeLevel::to_amp_scalar(volume_level);

        match noise_type {
            NoiseType::White => {
                let num_samples = frames * 2;
                for i in 0..num_samples {
                    buffer[i] = randf(&rng, -1f32, 1f32) * final_scalar;
                }
            }

            NoiseType::Pink => {
                let mut i = 0;
                for _ in 0..frames {
                    buffer[i] = generate_pink_noise(&rng, &mut left_pink) * final_scalar;
                    buffer[i + 1] = generate_pink_noise(&rng, &mut right_pink) * final_scalar;
                    i += 2
                }
            }

            NoiseType::Brown => {
                let mut i = 0;
                for _ in 0..frames {
                    buffer[i] = generate_brown_noise(&rng, &mut left_brown) * final_scalar;
                    buffer[i + 1] = generate_brown_noise(&rng, &mut right_brown) * final_scalar;
                    i += 2
                }
            }
        }

        pa::Continue
    };

    let mut stream = pa.open_non_blocking_stream(settings, callback)?;

    stream.start()?;

    let mut cleanup_gpio = None;

    match config.enable_gpio {
        true => cleanup_gpio = Some(setup_gpio(tx.clone())),
        false => (),
    };

    println!("generating noise. enter commands or type 'quit' to quit.");

    loop {
        let mut input = String::new();

        io::stdin().read_line(&mut input).unwrap();

        let msg = match input.trim() {
            "white" => Right(Message::SetNoiseType(NoiseType::White)),
            "pink" => Right(Message::SetNoiseType(NoiseType::Pink)),
            "brown" => Right(Message::SetNoiseType(NoiseType::Brown)),
            "next" => Right(Message::NextNoiseType),
            "up" => Right(Message::IncVolume),
            "down" => Right(Message::DecVolume),
            "vmax" => Right(Message::SetVolume(VolumeLevel::Max)),
            "vmin" => Right(Message::SetVolume(VolumeLevel::Mute)),
            "" | "exit" | "quit" => Left(None),
            inp => Left(Some(inp)),
        };

        match msg {
            Right(cmd) => tx.send(cmd).unwrap(),
            Left(Some(inp)) => println!("invalid input: {}", inp),
            Left(None) => break,
        };
    }

    stream.stop()?;
    stream.close()?;

    cleanup_gpio.map(|x| x());

    Ok(())
}
