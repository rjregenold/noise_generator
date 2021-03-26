#![allow(dead_code)]

use crossbeam_channel::tick;
use rppal::gpio::{Gpio, InputPin, Level, Trigger};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

const CHECK_MS: u64 = 5;
const PRESS_MS: u64 = 10;
const RELEASE_MS: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PushButtonBehavior {
    /// acts like a normal button (press/release)
    Switch,
    
    /// sends multiple pressed events when the button
    /// is held down.
    PushHold(Duration, Option<Duration>),
}

struct PinConfig {
    pin: u8,
    behavior: PushButtonBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ButtonState {
    Up,
    Down,
}

#[derive(Debug)]
struct PushButton {
    pin: InputPin,
    behavior: PushButtonBehavior,
    state: ButtonState,
    history: u8,
    count: u64,
    pushed_at: Option<Instant>,
    last_message_at: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    ButtonPushed(u8),
    ButtonHeld(u8),
    ButtonReleased(u8),
    ButtonHeldReleased(u8),
    Terminate,
}

/// a debounce function that looks for 8 of the same values
/// before considering the state stable. based on code found here:
/// https://git.sr.ht/~caolan/bramley/tree/4732f7af2fcec99d25797f78c089c547259238f8/buttons/src/lib.rs
fn debounce_switch0(btn: &mut PushButton) -> bool {
    let mut stable = false;
    let level = btn.pin.read();

    let changed = match level {
        Level::High => btn.state == ButtonState::Down,
        Level::Low => btn.state == ButtonState::Up,
    };

    if changed {
        stable = btn.history == 0 || btn.history == u8::MAX;
        if stable {
            let new_state = match level {
                Level::High => ButtonState::Up,
                Level::Low => ButtonState::Down,
            };

            btn.state = new_state;

            return true;
        }
    }

    btn.history = match level {
        Level::High => btn.history.rotate_left(1) | 0b00000001,
        Level::Low => btn.history.rotate_left(1) & 0b11111110,
    };

    stable
}

/// a time based debounce function. if a switch is stable for the 
/// configured amount of time, then that state can be used.
fn debounce_switch1(btn: &mut PushButton) -> bool {
    let level = btn.pin.read();
    let changed = match level {
        Level::High => btn.state == ButtonState::Down,
        Level::Low => btn.state == ButtonState::Up,
    };

    match changed {
        false => {
            btn.count = match btn.state {
                ButtonState::Down => RELEASE_MS / CHECK_MS,
                ButtonState::Up => PRESS_MS / CHECK_MS,
            };
        },
        true => {
            btn.count = btn.count - 1;

            if btn.count == 0 {
                btn.state = match level {
                    Level::High => ButtonState::Up,
                    Level::Low => ButtonState::Down,
                };

                btn.count = match btn.state {
                    ButtonState::Down => RELEASE_MS / CHECK_MS,
                    ButtonState::Up => PRESS_MS / CHECK_MS,
                };

                return true;
            }
        },
    };

    false
}

/// a function that performs no debouncing.
fn no_debounce_switch(btn: &mut PushButton) -> bool {
    let level = btn.pin.read();
    let changed = match (level, btn.state) {
        (Level::High, ButtonState::Down) => true,
        (Level::Low, ButtonState::Up) => true,
        _ => false,
    };

    match changed {
        true => btn.state = match btn.state {
            ButtonState::Down => ButtonState::Up,
            ButtonState::Up => ButtonState::Down,
        },
        _ => {}
    };

    changed
}

/// configures the provided pins as input pins and polls them
/// continuously to see if their level changes. if it does, a
/// message is broadcast on the returned receiver that contains
/// details about the button press/release. the returned closure
/// can be called to stop the polling.
pub fn setup_push_buttons_poll(pins: Vec<(u8, PushButtonBehavior)>) -> (mpsc::Receiver<Message>, impl FnOnce()) {
    let (term_tx, term_rx) = mpsc::channel();

    let clear_push_buttons = move || {
        term_tx.send(()).unwrap();
    };

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let gpio = Gpio::new().unwrap();

        let mut xs: Vec<PushButton> = pins.iter().map(|(pin, behavior)| PushButton {
            pin: gpio.get(*pin).unwrap().into_input_pullup(), 
            behavior: *behavior,
            state: ButtonState::Up, 
            history: 0,
            count: 0,
            pushed_at: None,
            last_message_at: None,
        }).collect();

        let timer = tick(Duration::from_millis(CHECK_MS));

        loop {
            for mut x in &mut xs {
                match no_debounce_switch(&mut x) {
                    true => {
                        let msg = match x.state {
                            ButtonState::Up => match x.last_message_at {
                                Some(_) => Message::ButtonHeldReleased(x.pin.pin()),
                                None => Message::ButtonReleased(x.pin.pin()),
                            },
                            ButtonState::Down => Message::ButtonPushed(x.pin.pin()),
                        };

                        if let Err(_err) = tx.send(msg) {
                            break;
                        }

                        match (x.behavior, x.state) {
                            (PushButtonBehavior::PushHold(_, _), ButtonState::Down) => {
                                x.pushed_at = Some(Instant::now());
                                x.last_message_at = None;
                            },
                            (PushButtonBehavior::PushHold(_, _), ButtonState::Up) => {
                                x.pushed_at = None;
                                x.last_message_at = None;
                            },
                            _ => {},
                        };
                    },
                    false => {
                        match (x.behavior, x.state) {
                            (PushButtonBehavior::PushHold(wait, interval), ButtonState::Down) => {
                                match (x.pushed_at, x.last_message_at) {
                                    (Some(pushed_at), None) if pushed_at.elapsed() >= wait => {
                                        if let Err(_err) = tx.send(Message::ButtonHeld(x.pin.pin())) {
                                            break;
                                        }

                                        x.last_message_at = Some(Instant::now());
                                    },
                                    (Some(_), Some(last_message_at)) if interval.map_or(false, |i| last_message_at.elapsed() >= i) => {
                                        if let Err(_err) = tx.send(Message::ButtonHeld(x.pin.pin())) {
                                            break;
                                        }

                                        x.last_message_at = Some(Instant::now());
                                    },
                                    _ => {},
                                }
                            },
                            _ => {},
                        };
                    },
                };
            }

            if let Err(_) = timer.recv() {
                // the timer stopped.
                tx.send(Message::Terminate).unwrap();
                break;
            }

            match term_rx.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => {
                    tx.send(Message::Terminate).unwrap();
                    break
                },
                Err(TryRecvError::Empty) => {},
            };
        }
    });

    (rx, clear_push_buttons)
}

/// configures the provided pins as input pins and configures 
/// interrupt callbacks to detect a button press/release. when
/// a button is pressed/released a message is broadcast on the 
/// returned receiver that contains details about the action. the 
/// returned closure can be called to clear the interrupts.
pub fn setup_push_buttons_interrupt(pins: Vec<u8>) -> (mpsc::Receiver<Message>, impl FnOnce()) {
    let (tx, rx) = mpsc::channel();

    let gpio = Gpio::new().unwrap();

    let mut input_pins: Vec<InputPin> = Vec::new();
    for pin in pins {
        let mut input_pin = gpio.get(pin).unwrap().into_input_pullup();

        let pin_tx = tx.clone();
        input_pin.set_async_interrupt(Trigger::Both, move |level| {
            let msg = match level {
                Level::High => Message::ButtonReleased(pin),
                Level::Low => Message::ButtonPushed(pin),
            };
            pin_tx.send(msg).unwrap();
        }).unwrap();

        input_pins.push(input_pin);
    }

    let clear_push_buttons = move || {
        for mut pin in input_pins {
            pin.clear_async_interrupt().unwrap();
        }
    };

    (rx, clear_push_buttons)
}