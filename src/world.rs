use std::sync::Arc;
use std::thread::available_parallelism;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};
use crate::vector::Vector;

pub struct World {
    pub particles: Vec<Particle>
}

impl World {
    pub fn new() -> Self {
        Self { particles: Vec::new() }
    }

    pub fn tick(&mut self, time: f64) {
        let particles_len = self.particles.len();
        let mut accelerations = vec![Vector::new(0.0, 0.0); particles_len];
        for i in 0..particles_len {
            for j in i+1..particles_len {
                let a = self.particles[i];
                let b = self.particles[j];
                let r_sq = Vector::distance_sq(&a.position, &b.position);
                // Newtons law of universal gravitation: (G * m1 * m2) / r^2
                let mut f = (6.674_30e-11 * a.mass * b.mass / r_sq) * time;
                if f.is_infinite() {
                    f = 0.0;
                }
                {
                    let (x1, y1) = a.position.to_cartesian();
                    let (x2, y2) = b.position.to_cartesian();
                    let d1 = f64::atan2(y2 - y1, x2 - x1);
                    let d2 = f64::atan2(y1 - y2, x1 - x2);
                    // f = ma
                    accelerations[i] += Vector::new(f / a.mass, d1);
                    accelerations[j] += Vector::new(f / b.mass, d2);
                }
            }
        }
        for (i, particle) in self.particles.iter_mut().enumerate() {
            particle.velocity.step(&accelerations[i], time);
            particle.position.step(&particle.velocity, time);
        }
    }

    pub fn get_mass_points(&self) -> Vec<MassPoint> {
        let mut mass_points = Vec::with_capacity(self.particles.len());
        for particle in &self.particles {
            mass_points.push(MassPoint {
                mass: particle.mass,
                position: particle.position.to_cartesian()
            })
        }
        mass_points
    }

    pub fn par_tick(&mut self, time: f64) {
        let particles = Arc::new(self.particles.clone());
        let accelerations = Self::tick_split(particles, 0, self.particles.len(), time);
        self.particles.par_iter_mut()
            .zip(accelerations)
            .for_each(|(particle, acceleration)| {
                particle.velocity.step(&acceleration, time);
                particle.position.step(&particle.velocity, time);
            });
    }

    fn tick_split(particles: Arc<Vec<Particle>>, lo: usize, hi: usize, time: f64) -> Vec<Vector> {
        let mid = (lo + hi) / 2;
        if mid == lo {
            let mut accelerations = vec![Vector::new(0.0, 0.0); particles.len()];
            let i = lo;
            for j in i + 1..particles.len() {
                let a = particles[i];
                let b = particles[j];
                let r_sq = Vector::distance_sq(&a.position, &b.position);
                // Newtons law of universal gravitation: (G * m1 * m2) / r^2
                let mut f = (6.674_30e-11 * a.mass * b.mass / r_sq) * time;
                if f.is_infinite() {
                    f = 0.0;
                }
                {
                    let (x1, y1) = a.position.to_cartesian();
                    let (x2, y2) = b.position.to_cartesian();
                    let d1 = f64::atan2(y2 - y1, x2 - x1);
                    let d2 = f64::atan2(y1 - y2, x1 - x2);
                    // f = ma
                    accelerations[i] += Vector::new(f / a.mass, d1);
                    accelerations[j] += Vector::new(f / b.mass, d2);
                }
            }
            accelerations
        } else {
            let particles_lo = particles.clone();
            let (lo, hi) = rayon::join(
                || Self::tick_split(particles_lo, lo, mid, time),
                || Self::tick_split(particles, mid, hi, time)
            );
            lo.into_iter()
                .zip(hi)
                .map(|(a, b)| {
                    a + b
                })
                .collect()
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MassPoint {
    pub mass: f64,
    pub position: (f64, f64)
}

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub mass: f64,
    pub position: Vector,
    pub velocity: Vector
}