use rppal::gpio::{Gpio, InputPin, Level};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ButtonState {
    Up,
    Down,
}

#[derive(Debug)]
struct PushButton {
    pin: InputPin,
    state: ButtonState,
    history: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    ButtonPressed(u8),
    ButtonReleased(u8),
    Terminate,
}

pub fn setup_push_buttons(pins: Vec<u8>) -> (mpsc::Receiver<Message>, impl FnOnce()) {
    // setting up async interrupts resulted in too much bounce
    // that was too unpredictable to reliably handle with debouncing.
    // instead, we poll the requested pins and wait for 8 of the same
    // values in a row before considering the signal stable. this is
    // based on code found here:
    // https://git.sr.ht/~caolan/bramley/tree/4732f7af2fcec99d25797f78c089c547259238f8/buttons/src/lib.rs

    let (term_tx, term_rx) = mpsc::channel();

    let clear_push_buttons = move || {
        term_tx.send(()).unwrap();
    };

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let gpio = Gpio::new().unwrap();

        let mut xs: Vec<PushButton> = pins.iter().map(|pin| PushButton {
            pin: gpio.get(*pin).unwrap().into_input_pullup(), 
            state: ButtonState::Up, 
            history: 0u8,
        }).collect();

        loop {
            for mut x in &mut xs {
                let level = x.pin.read();

                let changed = match level {
                    Level::High => x.state == ButtonState::Down,
                    Level::Low => x.state == ButtonState::Up,
                };

                if changed {
                    let stable = x.history == 0 || x.history == u8::MAX;
                    if stable {
                        let new_state = match level {
                            Level::High => ButtonState::Up,
                            Level::Low => ButtonState::Down,
                        };

                        x.state = new_state;

                        let msg = match x.state {
                            ButtonState::Up => Message::ButtonReleased(x.pin.pin()),
                            ButtonState::Down => Message::ButtonPressed(x.pin.pin()),
                        };

                        if let Err(_err) = tx.send(msg) {
                            break;
                        }
                    }
                }

                x.history = match level {
                    Level::High => x.history.rotate_left(1) | 0b00000001,
                    Level::Low => x.history.rotate_left(1) & 0b11111110,
                };
            }

            thread::sleep(Duration::from_millis(3));

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