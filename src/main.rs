use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::fs::File;
use std::num::{NonZeroU16, NonZeroUsize};
use std::ops::Range;
use std::{fs, thread};
use std::cmp::Ordering;
use std::path::Path;
use std::thread::available_parallelism;
use std::time::Instant;
use image::codecs::gif::{GifDecoder, GifEncoder, Repeat};
use image::{AnimationDecoder, DynamicImage, Frame, GenericImage, ImageBuffer, open, Pixel, Rgb, RgbaImage, RgbImage};
use image::io::Reader;
use imageproc::drawing::draw_filled_circle_mut;
use rand::{Rng, SeedableRng};
use log::{Level, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Logger, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::Config;
use num_traits::ToPrimitive;
use rand_pcg::Pcg64Mcg;
use rayon::ThreadPoolBuilder;
use world::cpu::CPUWorld;
use world::gpu::GPUWorld;
use crate::periodic_logger::PeriodicLogger;
use crate::render::cpu::{AreaIntersectionRasterizer, FastIntegerRasterizer, GrayscaleRgbScalar, HorizontalLineImage, Rasterizer};
use crate::vector::Vector;
use crate::world::{MassPoint, Particle};
use crate::world::par::ParWorld;

mod vector;
mod periodic_logger;
mod world;
mod render;

const SEED: u64 = 23;
const PARTICLE_COUNT: usize = 100;
const FRAME_COUNT: usize = 240;
const SCALE: f32 = 500.0;
const TIME_PER_FRAME: f32 = 20.0;
const TIME_STEPS: NonZeroU16 = match NonZeroU16::new(20) {
    None => panic!("TIME_STEPS may not be 0"),
    Some(steps) => steps
};
const SIZE: Option<(f32, f32)> = Some((1000.0, 1000.0));
const PARTICLE_GENERATOR: fn() -> Vec<Particle> = generate_particles;

fn main() {
    initialize_logging();
    let circles = [
        (50.0, 50.0, 25.0),
        (11.0, 11.0, 10.0),
        (11.0, 33.0, 10.5),
        (2.0, 46.0, 1.0),
        (5.0, 46.0, 0.5),
        (8.5, 46.5, 1.0),
        (11.5, 46.5, 0.5),
        (14.5, 46.5, 0.7),
        (17.5, 46.5, 0.3),
        (14.0, 86.3, 13.0),
        (86.5, 86.5, 13.0)
    ];

    let mut image: HorizontalLineImage<_, _> = RgbImage::new(100, 100).into();
    //let mut image = RgbImage::new(100 * SCALE, 100 * SCALE);
    for (cx, cy, r) in circles {
        //<FastIntegerRasterizer as Rasterizer<_, _, GrayscaleRgbScalar>>::draw_filled_circle(&mut image, cx as f32, cy as f32, r as f32, [255, 255, 255].into());
        <AreaIntersectionRasterizer as Rasterizer<_, _, GrayscaleRgbScalar>>::draw_filled_circle(&mut image, cx, cy, r, [255, 255, 255].into());

        /*
        approximate_area_intersection_draw_circle::<SUB_DIV, _>(|x, y, f| {
            let c = (f * 255.0) as u8;
            unsafe {
                image.unsafe_put_pixel(x, y, [c; 3].into())
            }
        }, cx * SCALE as f32, cy * SCALE as f32, r * SCALE as f32);
         */
    }
    println!("overriding image");
    RgbImage::from(image).save("output/circle.png").unwrap();
    let (image_diff, num_diff) = rgb_image_subtract("output/inter_circle.png", "output/circle.png", 1.0);
    image_diff.save("output/difference.png").unwrap();
    println!("difference: {}", num_diff);

    for _ in 0..10 {
        const SCALE: u32 = 128;
        let start = Instant::now();
        let mut image: HorizontalLineImage<_, _> = RgbImage::new(100 * SCALE, 100 * SCALE).into();
        //let mut image = RgbImage::new(100 * SCALE, 100 * SCALE);
        const SUB_DIV: usize = 3;
        for (cx, cy, r) in circles {
            //<FastIntegerRasterizer as Rasterizer<_, _, GrayscaleRgbScalar>>::draw_filled_circle(&mut image, cx * SCALE as f32, cy * SCALE as f32, r * SCALE as f32, [255, 255, 255].into());
            <AreaIntersectionRasterizer as Rasterizer<_, _, GrayscaleRgbScalar>>::draw_filled_circle(&mut image, cx * SCALE as f32, cy * SCALE as f32, r * SCALE as f32, [255, 255, 255].into());

            /*
            approximate_area_intersection_draw_circle::<SUB_DIV, _>(|x, y, f| {
                let c = (f * 255.0) as u8;
                unsafe {
                    image.unsafe_put_pixel(x, y, [c; 3].into())
                }
            }, cx * SCALE as f32, cy * SCALE as f32, r * SCALE as f32);
             */
        }
        println!("{:02.3}", start.elapsed().as_secs_f32());
    }
}

fn rgb_image_subtract<P: AsRef<Path>>(path_a: P, path_b: P, amplifier: f32) -> (RgbImage, u64) {
    let image_a = if let DynamicImage::ImageRgb8(image) =
        Reader::open(path_a).unwrap().decode().unwrap()
            { image } else { panic!() };
    let image_b = if let DynamicImage::ImageRgb8(image) =
        Reader::open(path_b).unwrap().decode().unwrap()
            { image } else { panic!() };
    assert_eq!(image_a.width(), image_b.width());
    let width = image_a.width();
    assert_eq!(image_a.height(), image_b.height());
    let height = image_a.height();
    let mut image_diff = RgbImage::new(width, height);
    let mut diff = 0;
    for ((a, b), c) in image_a.pixels().zip(image_b.pixels()).zip(image_diff.pixels_mut()) {
        let a = a.0[0];
        let b = b.0[0];
        match a.cmp(&b) {
            Ordering::Less => {
                c.0 = [((b as f32 - a as f32) * amplifier) as u8, ((b as f32 - a as f32) * amplifier) as u8, 0];
                diff += (b - a) as u64;
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                c.0 = [0, 0, ((a as f32 - b as f32) * amplifier) as u8];
                diff += (a - b) as u64;
            }
        }
    }
    (image_diff, diff)
}

fn approximate_area_intersection_draw_circle<const SUB_DIV: usize, F: FnMut(u32, u32, f32)>(mut put_pixel: F, cx: f32, cy: f32, r: f32) {
    let min_x = (cx - r) as u32;
    let max_x = (cx + r + 1.0) as u32;
    let min_y = (cy - r) as u32;
    let max_y = (cy + r + 1.0) as u32;
    let max_inside_count = (SUB_DIV * SUB_DIV) as f32;
    let r_sq = r * r;
    for y in min_y..max_y {
        let ys = sub_divisions::<SUB_DIV>(y);
        let mut y_diffs_sq = ys;
        for y in y_diffs_sq.iter_mut() {
            *y -= cy;
            *y *= *y;
        }
        for x in min_x..max_x {
            let xs = sub_divisions::<SUB_DIV>(x);
            let mut inside_count = 0u32;
            for y_i in 0..SUB_DIV {
                for x in xs {
                    let x_diff = cx - x;
                    let distance_sq = x_diff * x_diff + y_diffs_sq[y_i];
                    if distance_sq <= r_sq {
                        inside_count += 1;
                    }
                }
            }
            let c = inside_count as f32 / max_inside_count;
            put_pixel(x, y, c);
        }
    }
}

#[inline(always)]
fn sub_divisions<const SUB_DIV: usize>(v: u32) -> [f32; SUB_DIV] {
    let mut vs = [0.0; SUB_DIV];
    if SUB_DIV > 0 {
        let diff = 1.0 / (SUB_DIV + 1) as f32;
        vs[0] = v as f32 + diff;
        for i in 1..SUB_DIV {
            vs[i] = vs[i-1] + diff;
        }
    }
    vs
}

fn draw_circle<F: FnMut(u32, u32, f32)>(mut put_pixel: F, cx: f32, cy: f32, r: f32) {
    let mut y = 0.0;
    let mut x = r;
    let mut d = 0.0;
    mirror_put_pixel(&mut put_pixel, cx, cy, x, y, 1.0);
    while x > y {
        y += 1.0;
        let dtc = distance_to_ceil(r, y);
        if dtc < d {
            x -= 1.0;
        }
        d = dtc;
        mirror_put_pixel(&mut put_pixel, cx, cy, x, y, 1.0 - d);
        mirror_put_pixel(&mut put_pixel, cx, cy, x - 1.0, y, d);
    }
}

#[inline(always)]
fn mirror_put_pixel<F: FnMut(u32, u32, f32)>(put_pixel: &mut F, c_x: f32, c_y: f32, x: f32, y: f32, c: f32) {
    let min_x = (c_x - x).to_u32();
    let max_x = (c_x + x).to_u32();
    let min_y = (c_y - y).to_u32();
    let max_y = (c_y + y).to_u32();
    mirror_put_pixel_if_some(put_pixel, min_x, min_y, c);
    mirror_put_pixel_if_some(put_pixel, max_x, min_y, c);
    mirror_put_pixel_if_some(put_pixel, min_x, max_y, c);
    mirror_put_pixel_if_some(put_pixel, max_x, max_y, c);
}

#[inline(always)]
fn mirror_put_pixel_if_some<F: FnMut(u32, u32, f32)>(put_pixel: &mut F, x: Option<u32>, y: Option<u32>, c: f32) {
    match (x, y) {
        (Some(x), Some(y)) => {
            put_pixel(x, y, c);
            put_pixel(y, x, c);
        },
        _ => {}
    }
}

#[inline(always)]
fn distance_to_ceil(r: f32, y: f32) -> f32 {
    let x = f32::sqrt(r * r - y * y);
    f32::ceil(x) - x
}

#[allow(dead_code)]
fn output_gpu() {
    let world = GPUWorld::new(PARTICLE_GENERATOR());
    tick_and_output_gif(world, GPUWorld::tick, GPUWorld::get_mass_points, "gpu");
}

#[allow(dead_code)]
fn compare_outputs() {
    let particles = PARTICLE_GENERATOR();
    let particles_a = particles.clone();
    let particles_b = particles.clone();
    let particles_c = particles;

    ThreadPoolBuilder::new()
        .num_threads(
            usize::max(
                available_parallelism()
                    .unwrap_or(NonZeroUsize::new(1).unwrap())
                    .get() - 1,
                1)
        )
        .build_global()
        .unwrap();
    let handles = [
        thread::spawn(|| {
            let world = CPUWorld { particles: particles_a };
            tick_and_output_gif(world, CPUWorld::tick, CPUWorld::get_mass_points, "cpu");
        }),
        thread::spawn(|| {
            let world = ParWorld::new(particles_b);
            tick_and_output_gif(world, ParWorld::tick, ParWorld::get_mass_points, "par");
        }),
        thread::spawn(|| {
            let world = GPUWorld::new(particles_c);
            tick_and_output_gif(world, GPUWorld::tick, GPUWorld::get_mass_points, "gpu");
        })
    ];
    for handle in handles {
        handle.join().unwrap();
    }

    {
        let single = GifDecoder::new(File::open("output/cpu.gif").unwrap()).unwrap();
        let multi = GifDecoder::new(File::open("output/par.gif").unwrap()).unwrap();
        let gpu = GifDecoder::new(File::open("output/gpu.gif").unwrap()).unwrap();
        let mut merged = GifEncoder::new(File::create("output/merged.gif").unwrap());
        merged.set_repeat(Repeat::Infinite).unwrap();
        let mut periodic_logger = PeriodicLogger::new("exporting merged", Level::Info);
        single.into_frames()
            .zip(multi.into_frames())
            .zip(gpu.into_frames())
            .map(|((single_frame_result, multi_frame_result), gpu_frame_result)| {
                (single_frame_result.unwrap().into_buffer(), multi_frame_result.unwrap().into_buffer(), gpu_frame_result.unwrap().into_buffer())
            })
            .enumerate()
            .for_each(|(frame, (single_frame, multi_frame, gpu_frame))| {
                periodic_logger.log(format!("{} / {}", frame, FRAME_COUNT));
                let (width, height) = (single_frame.width(), single_frame.height());
                let mut image = RgbaImage::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        let r = single_frame[(x, y)].0[0];
                        let g = multi_frame[(x, y)].0[1];
                        let b = gpu_frame[(x, y)].0[2];
                        image[(x, y)].0 = [r, g, b, 255];
                    }
                }
                merged.encode_frame(Frame::new(image)).unwrap();
            });
    }
}

#[allow(dead_code)]
fn generate_particles() -> Vec<Particle> {
    let mut rng = Pcg64Mcg::seed_from_u64(SEED);
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    for _ in 0..PARTICLE_COUNT {
        particles.push(Particle {
            mass: rng.gen_range(0.0..1.0),
            position: Vector::new(rng.gen_range(0.0..TAU), rng.gen_range(0.5..1.0)),
            velocity: Vector::new(0.0, 0.0)
        });
    }
    particles
}

#[allow(dead_code)]
fn generate_3_body() -> Vec<Particle> {
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    particles.push(Particle {
        mass: 10000.0,
        position: Vector::new(0.0, 0.0),
        velocity: Vector::new(0.0, 0.0)
    });
    particles.push(Particle {
        mass: 100.0,
        position: Vector::new(0.0, 0.50),
        velocity: Vector::new(FRAC_PI_2, 0.001)
    });
    particles.push(Particle {
        mass: 10.0,
        position: Vector::new(0.0, 0.55),
        velocity: Vector::new(FRAC_PI_2, 0.0013)
    });
    particles
}

fn tick_and_output_gif<W, TF: FnMut(&mut W, f32, NonZeroU16), MPG: FnMut(&W) -> Vec<MassPoint>>(mut world: W, mut tick_function: TF, mut mass_point_getter: MPG, name: &str) {
    let mut periodic_logger = PeriodicLogger::new(&format!("simulating {}", name), Level::Info);
    let mut mass_position_frames = Vec::with_capacity(FRAME_COUNT);
    for frame in 0..FRAME_COUNT {
        tick_function(&mut world, TIME_PER_FRAME, TIME_STEPS);
        mass_position_frames.push(mass_point_getter(&world));
        periodic_logger.log(format!("{} / {}", frame, FRAME_COUNT));
    }
    output_gif(mass_position_frames, name);
}

fn output_gif(mass_position_frames: Vec<Vec<MassPoint>>, name: &str) {
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
        File::create(format!("output/{}.gif", name))
            .expect("unable to create file")
    );
    gif.set_repeat(Repeat::Infinite)
        .expect("unable to make gif infinitely repeatable");
    let mut periodic_logger = PeriodicLogger::new(&format!("exporting {}", name), Level::Info);
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
                f32::cbrt(3.0 * mass / 4.0 * PI) as i32,
                [255, 255, 255, 255].into()
            );
        }
        gif.encode_frame(Frame::new(image))
            .expect("error occurred while encoding frame");
        periodic_logger.log(format!("{} / {}", frame, FRAME_COUNT));
    }
}

fn adjust_bounds(bounds: &mut Range<f32>, v: f32) {
    if v < bounds.start {
        bounds.start = v;
    } else if v > bounds.end {
        bounds.end = v;
    }
}

#[deny(dead_code)]
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
