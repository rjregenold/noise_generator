extern crate portaudio;
extern crate clap;

mod gpio;

use clap::{Arg, App};
use either::*;
use fastrand;
use portaudio as pa;
use std::io;
use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const CHANNELS: i32 = 2;
const FADE_IN_SECS: f32 = 10f32;
const FADE_OUT_SECS: f32 = 2f32;

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
    perform_shutdown: bool,
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
                        .arg(Arg::with_name("perform-shutdown")
                            .short("s")
                            .long("perform-shutdown")
                            .help("Executes shutdown command when done"))
                        .arg(Arg::with_name("settings")
                            .short("s")
                            .long("settings")
                            .help("Path to file where app settings can be stored")
                            .takes_value(true))
                        .get_matches();

        let enable_gpio = matches.is_present("enable-gpio");
        let perform_shutdown = matches.is_present("perform-shutdown");
        let settings_path = matches.value_of("settings");

        Config { enable_gpio, perform_shutdown, settings_path: settings_path.map(String::from) }
    }
}

/// Generates a random f32 between low and high.
fn randf(rng: &fastrand::Rng, low: f32, high: f32) -> f32 {
    (rng.u16(..) as f32 / RAND_MAX) * f32::abs(low - high) + low
}

#[derive(Clone, Copy, Debug)]
enum VolumeLevel {
    // do not allow volume to be muted as the user might
    // not know if the machine is off or just muted.
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
            VolumeLevel::_1 => VolumeLevel::_1,
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
    PowerPushed,
    PowerHeld,
    SetNoiseType(NoiseType),
    NextNoiseType,
    SetVolume(VolumeLevel),
    DecVolume,
    IncVolume,
}

#[derive(Clone, Copy, Debug)]
enum StreamMessage {
    Start(NoiseType, VolumeLevel),
    Stop,
    SetNoiseType(NoiseType),
    SetVolume(VolumeLevel),
}

fn main() {
    match run(Config::new()) {
        Ok(_) => {}
        e => {
            eprintln!("Failed to run: {:?}", e);
        }
    }
}

const PIN_POWER: u8 = 3;
const PIN_NOISE_TYPE: u8 = 22;
const PIN_INC_VOLUME: u8 = 23;
const PIN_DEC_VOLUME: u8 = 24;

fn setup_gpio(tx: mpsc::Sender<Message>) -> impl FnOnce() {
    let (rx, clear_push_buttons) = gpio::setup_push_buttons_poll(vec![
        (PIN_POWER, gpio::PushButtonBehavior::PushHold(Duration::from_secs(2), None)),
        (PIN_NOISE_TYPE, gpio::PushButtonBehavior::Switch),
        (PIN_INC_VOLUME, gpio::PushButtonBehavior::PushHold(Duration::from_secs(1), Some(Duration::from_millis(250)))),
        (PIN_DEC_VOLUME, gpio::PushButtonBehavior::PushHold(Duration::from_secs(1), Some(Duration::from_millis(250)))),
    ]);

    let cleanup = move || {
        clear_push_buttons();
    };

    std::thread::spawn(move || {
        loop {
            for msg in &rx {
                match msg {
                    gpio::Message::ButtonPushed(pin) => {
                        match pin {
                            PIN_NOISE_TYPE => tx.send(Message::NextNoiseType).unwrap(),
                            PIN_INC_VOLUME => tx.send(Message::IncVolume).unwrap(),
                            PIN_DEC_VOLUME => tx.send(Message::DecVolume).unwrap(),
                            _ => {},
                        };
                    },
                    gpio::Message::ButtonHeld(pin) => {
                        match pin {
                            PIN_INC_VOLUME => tx.send(Message::IncVolume).unwrap(),
                            PIN_DEC_VOLUME => tx.send(Message::DecVolume).unwrap(),
                            _ => {},
                        }
                    },
                    gpio::Message::ButtonReleased(pin) => {
                        match pin {
                            PIN_POWER => tx.send(Message::PowerPushed).unwrap(),
                            _ => {},
                        }
                    },
                    gpio::Message::ButtonHeldReleased(pin) => {
                        match pin {
                            PIN_POWER => tx.send(Message::PowerHeld).unwrap(),
                            _ => {},
                        }
                    }
                    gpio::Message::Terminate => break,
                };
            }
        }
    });

    cleanup
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum StreamState {
    Stopped,
    Starting,
    Running,
    Stopping,
}

fn setup_stream(rx: mpsc::Receiver<StreamMessage>, init_noise_type: NoiseType, init_volume_level: VolumeLevel) -> Result<pa::Stream<pa::NonBlocking, pa::Output<f32>>, pa::Error> {
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
    let mut started_at = None;

    let mut left_pink = init_pink_noise(12);
    let mut right_pink = init_pink_noise(16);

    let mut left_brown = init_brown_noise(&rng);
    let mut right_brown = init_brown_noise(&rng);

    let mut fade_out_start_at: Option<Instant> = None;

    let mut noise_type = init_noise_type;
    let mut volume_level = init_volume_level;
    let mut stream_state = StreamState::Starting;

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        for m in rx.try_iter() {
            match m {
                StreamMessage::Start(nt, l) => {
                    stream_state = StreamState::Starting;
                    noise_type = nt;
                    volume_level = l;
                    started_at = None;
                    fade_out_start_at = None;
                    fade_in_scalar = 0.0;
                },
                StreamMessage::Stop => {
                    stream_state = StreamState::Stopping;
                    fade_out_start_at = Some(Instant::now());
                },
                StreamMessage::SetNoiseType(nt) => noise_type = nt,
                StreamMessage::SetVolume(v) => volume_level = v,
            }
        }

        if stream_state == StreamState::Stopped {
            return pa::Complete;
        }

        if stream_state == StreamState::Starting {
            // there is a bug in the portaudio alsa api that makes
            // the `time` argument empty, so we use the system clock
            // https://github.com/PortAudio/portaudio/issues/498
            let now = started_at.get_or_insert(Instant::now());
            let elapsed = now.elapsed();
            let delta = elapsed.as_secs_f32();

            fade_in_scalar = ((delta / FADE_IN_SECS) + 1.0).log2().min(1.0);

            if fade_in_scalar >= 1f32 {
                fade_in_scalar = 1f32;
                stream_state = StreamState::Running;
            }
        }

        let mut fade_out_scalar = fade_out_start_at.map_or(1f32, |x| (FADE_OUT_SECS - x.elapsed().as_secs_f32()) / FADE_OUT_SECS);

        if fade_out_scalar <= 0f32 {
            fade_out_scalar = 0f32;
            stream_state = StreamState::Stopped;
        }

        let final_scalar = fade_in_scalar * VolumeLevel::to_amp_scalar(volume_level) * fade_out_scalar;

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

    pa.open_non_blocking_stream(settings, callback)
}

fn run(config: Config) -> Result<(), pa::Error> {
    let (tx, rx) = mpsc::channel();

    let mut cleanup_gpio = None;

    match config.enable_gpio {
        true => cleanup_gpio = Some(setup_gpio(tx.clone())),
        false => (),
    };

    println!("noise machine running. enter commands or type 'quit' to quit.");

    std::thread::spawn(move || {
        loop {
            let mut input = String::new();

            io::stdin().read_line(&mut input).unwrap();

            let msg = match input.trim() {
                "toggle" | "start" | "stop" => Right(Message::PowerPushed),
                "white" => Right(Message::SetNoiseType(NoiseType::White)),
                "pink" => Right(Message::SetNoiseType(NoiseType::Pink)),
                "brown" => Right(Message::SetNoiseType(NoiseType::Brown)),
                "next" => Right(Message::NextNoiseType),
                "up" => Right(Message::IncVolume),
                "down" => Right(Message::DecVolume),
                "vmax" => Right(Message::SetVolume(VolumeLevel::Max)),
                "vmin" => Right(Message::SetVolume(VolumeLevel::_1)),
                "" | "exit" | "quit" => Right(Message::PowerHeld),
                inp => Left(inp),
            };

            match msg {
                Right(cmd) => tx.send(cmd).unwrap(),
                Left(inp) => println!("invalid input: {}", inp),
            };
        }
    });

    // TODO: read these from a settings file.
    let mut noise_type: NoiseType = NoiseType::White;
    let mut volume_level: VolumeLevel = VolumeLevel::_5;

    let (stream_tx, stream_rx) = mpsc::channel();
    let mut stream = setup_stream(stream_rx, noise_type, volume_level).unwrap();

    'main_loop: loop {
        for m in rx.iter() {
            match m {
                Message::PowerPushed => {
                    if !stream.is_active().unwrap() {
                        stream_tx.send(StreamMessage::Start(noise_type, volume_level)).unwrap();

                        match stream.start() {
                            Ok(_) => {},
                            // if for some reason the stream was not stopped then
                            // stop it and start it again (this can happen if the
                            // power button is tapped twice really quickly).
                            Err(pa::Error::StreamIsNotStopped) => {
                                stream.stop().unwrap();

                                stream_tx.send(StreamMessage::Start(noise_type, volume_level)).unwrap();
                                stream.start().unwrap();
                            },
                            _ => {}
                        }
                    } else {
                        stream_tx.send(StreamMessage::Stop).unwrap();

                        // let the stream fade out and then stop it.
                        while let true = stream.is_active().unwrap() {
                            std::thread::sleep(Duration::from_millis(500));
                        }

                        stream.stop().unwrap();

                        // drain any messages that came in whilst waiting for
                        // the stream to stop.
                        let _ = rx.try_recv();
                    }
                },
                Message::PowerHeld => {
                    // gracefully end the stream if it is running
                    if stream.is_active().unwrap() {
                        stream_tx.send(StreamMessage::Stop).unwrap();

                        // let the stream fade out and then stop it.
                        while let true = stream.is_active().unwrap() {
                            std::thread::sleep(Duration::from_millis(500));
                        }

                        stream.stop().unwrap();

                        // drain any messages that came in whilst waiting for
                        // the stream to stop.
                        let _ = rx.try_recv();
                    }

                    if config.perform_shutdown {
                        println!("shutting down pi");

                        let output = Command::new("shutdown")
                            .args(&["-h", "now"])
                            .output()
                            .expect("Failed to shutdown pi");

                        println!("{:?}", output);
                    }

                    break 'main_loop
                },
                Message::SetNoiseType(nt) => {
                    noise_type = nt;
                    stream_tx.send(StreamMessage::SetNoiseType(noise_type)).unwrap();
                },
                Message::NextNoiseType => {
                    noise_type = NoiseType::succ(noise_type);
                    stream_tx.send(StreamMessage::SetNoiseType(noise_type)).unwrap();
                },
                Message::SetVolume(v) =>  {
                    volume_level = v;
                    stream_tx.send(StreamMessage::SetVolume(volume_level)).unwrap();
                },
                Message::IncVolume => {
                    volume_level = VolumeLevel::succ(volume_level);
                    stream_tx.send(StreamMessage::SetVolume(volume_level)).unwrap();
                },
                Message::DecVolume => {
                    volume_level = VolumeLevel::pred(volume_level);
                    stream_tx.send(StreamMessage::SetVolume(volume_level)).unwrap();
                },
            }
        }
    }

    cleanup_gpio.map(|x| x());

    Ok(())
}
