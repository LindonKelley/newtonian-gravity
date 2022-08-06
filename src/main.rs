use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fs::File;
use std::ops::Range;
use image::codecs::gif::{GifEncoder, Repeat};
use image::{Frame, RgbaImage};
use imageproc::drawing::draw_filled_circle_mut;
use rand::{Rng, thread_rng};
use log::{Level, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Logger, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::Config;
use crate::periodic_logger::PeriodicLogger;
use crate::vector::Vector;
use crate::world::{MassPoint, Particle, World};

mod vector;
mod world;
mod periodic_logger;

fn main() {
    initialize_logging();

    const PARTICLE_COUNT: usize = 1000;
    const FRAME_COUNT: u32 = 4800;
    const SCALE: f64 = 500.0;
    const TIME_SCALE: f64 = 1.0;
    // maybe switch over to quality steps
    const TIME_STEP: f64 = 1.0;
    const SIZE: Option<(f64, f64)> = Some((1000.0, 1000.0));

    let mut rng = thread_rng();
    let mut world = World::new();
    world.particles.push(Particle {
        mass: 10000.0,
        position: Vector::new(0.0, 0.0),
        velocity: Vector::new(0.0, 0.0)
    });
    world.particles.push(Particle {
        mass: 100.0,
        position: Vector::new(0.50, 0.0),
        velocity: Vector::new(0.001, FRAC_PI_2)
    });
    world.particles.push(Particle {
        mass: 10.0,
        position: Vector::new(0.55, 0.0),
        velocity: Vector::new(0.0013, FRAC_PI_2)
    });
    for _ in 0..PARTICLE_COUNT {
        world.particles.push(Particle {
            mass: rng.gen_range(0.0..1.0),
            position: Vector::new(rng.gen_range(0.5..1.0), rng.gen_range(0.0..TAU)),
            velocity: Vector::new(0.0, 0.0)
        });
    }

    let mut mass_position_frames = Vec::with_capacity(FRAME_COUNT as usize);
    mass_position_frames.push(world.get_mass_points());
    let mut time_scale_render = 0.0;
    let mut periodic_logger = PeriodicLogger::new("simulating", Level::Info);
    for frame in 0..FRAME_COUNT {
        while time_scale_render < TIME_SCALE {
            world.par_tick(TIME_STEP);
            time_scale_render += TIME_STEP;
        }
        mass_position_frames.push(world.get_mass_points());
        time_scale_render -= TIME_SCALE;
        periodic_logger.log(format!("{} / {}", frame, FRAME_COUNT));
    }

    let mut bounds_x;
    let mut bounds_y;
    let mut bounds_mass;
    if let Some((width, height)) = SIZE {
        let w = (width - 1.0) / 2.0 / SCALE;
        let h = (height - 1.0) / 2.0 / SCALE;
        bounds_x = -w..w;
        bounds_y = -h..h;
        bounds_mass = 0.0..1000.0;
    } else {
        {
            let MassPoint { mass, position: (x, y) } = mass_position_frames[0][0];
            bounds_x = x..x;
            bounds_y = y..y;
            bounds_mass = mass..mass;
        }
        for mass_positions in &mass_position_frames {
            for mass_position in mass_positions {
                let MassPoint { mass, position: (x, y) } = *mass_position;
                adjust_bounds(&mut bounds_mass, mass);
                adjust_bounds(&mut bounds_x, x);
                adjust_bounds(&mut bounds_y, y);
            }
        }
    }


    let width = ((bounds_x.end - bounds_x.start) * SCALE) as u32 + 1;
    let height = ((bounds_y.end - bounds_y.start) * SCALE) as u32 + 1;
    let mut gif = GifEncoder::new(
        File::create("output/gravity.gif")
            .expect("unable to create file")
    );
    gif.set_repeat(Repeat::Infinite)
        .expect("unable to make gif infinitely repeatable");
    let mut periodic_logger = PeriodicLogger::new("exporting", Level::Info);
    for (frame, mass_positions) in mass_position_frames.iter().enumerate() {
        let mut image = RgbaImage::new(width, height);
        for pixel in image.pixels_mut() {
            pixel.0 = [0, 0, 0, 255];
        }
        for mass_position in mass_positions {
            let MassPoint { mass, position: (x, y) } = mass_position;
            let px = ((x - bounds_x.start) * SCALE) as i32;
            let py = ((y - bounds_y.start) * SCALE) as i32;
            //let m = ((1.0 - mass / bounds_mass.end) * 255.0) as u8;
            draw_filled_circle_mut(
                &mut image,
                (px, py),
                f64::cbrt(3.0 * mass / 4.0 * PI) as i32,
                [255, 255, 255, 255].into()
            );
        }
        gif.encode_frame(Frame::new(image))
            .expect("error occurred while encoding frame");
        periodic_logger.log(format!("{} / {}", frame, FRAME_COUNT));
    }
}

fn adjust_bounds(bounds: &mut Range<f64>, v: f64) {
    if v < bounds.start {
        bounds.start = v;
    } else if v > bounds.end {
        bounds.end = v;
    }
}

fn initialize_logging() {
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{m}{n}")))
        .build();

    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .logger(Logger::builder().build("app::backend::db", LevelFilter::Info))
        .build(Root::builder().appender("stdout").build(LevelFilter::Info))
        .unwrap();

    log4rs::init_config(config).unwrap();
}
