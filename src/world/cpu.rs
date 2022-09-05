use std::num::NonZeroU16;
use std::f32::consts::PI;
use std::sync::Arc;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::{MassPoint, Particle, Vector};

pub struct CPUWorld {
    pub particles: Vec<Particle>
}

impl CPUWorld {
    pub fn new() -> Self {
        Self { particles: Vec::new() }
    }

    pub fn tick(&mut self, time: f32, steps: NonZeroU16) {
        let stepped_time = time / steps.get() as f32;
        for _ in 0..steps.get() {
            let particles_len = self.particles.len();
            let mut accelerations = vec![Vector::new(0.0, 0.0); particles_len];
            for i in 0..particles_len {
                for j in i + 1..particles_len {
                    let a = self.particles[i];
                    let b = self.particles[j];
                    let r_sq = Vector::distance_sq(&a.position, &b.position);
                    // Newtons law of universal gravitation: (G * m1 * m2) / r^2
                    let f = (6.67430e-11 * a.mass * b.mass / r_sq) * stepped_time;
                    if f.is_infinite() {
                        continue
                    } else {
                        let (x1, y1) = a.position.to_cartesian();
                        let (x2, y2) = b.position.to_cartesian();
                        let d1 = f32::atan2(y2 - y1, x2 - x1);
                        let d2 = d1 + PI;
                        // f = ma
                        accelerations[i] += Vector::new(d1, f / a.mass);
                        accelerations[j] += Vector::new(d2, f / b.mass);
                    }
                }
            }
            for (i, particle) in self.particles.iter_mut().enumerate() {
                particle.velocity.step(&accelerations[i], stepped_time);
                particle.position.step(&particle.velocity, stepped_time);
            }
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
}
