use std::f32::consts::PI;
use std::num::NonZeroU16;
use std::sync::Arc;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::{MassPoint, Particle, Vector};

pub struct ParWorld {
    particles: Arc<Vec<Particle>>
}

impl ParWorld {
    pub fn new(particles: Vec<Particle>) -> Self {
        Self {
            particles: Arc::new(particles)
        }
    }

    pub fn tick(&mut self, time: f32, steps: NonZeroU16) {
        let stepped_time = time / steps.get() as f32;
        for _ in 0..steps.get() {
            let accelerations = Self::tick_split(self.particles.clone(), 0, self.particles.len(), stepped_time);
            Arc::get_mut(&mut self.particles).unwrap().par_iter_mut()
                .zip(accelerations)
                .for_each(|(particle, acceleration)| {
                    particle.velocity.step(&acceleration, stepped_time);
                    particle.position.step(&particle.velocity, stepped_time);
                });
        }
    }

    fn tick_split(particles: Arc<Vec<Particle>>, lo: usize, hi: usize, stepped_time: f32) -> Vec<Vector> {
        let mid = (lo + hi) / 2;
        if mid == lo {
            let mut accelerations = vec![Vector::new(0.0, 0.0); particles.len()];
            let i = lo;
            for j in i + 1..particles.len() {
                let a = particles[i];
                let b = particles[j];
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
            accelerations
        } else {
            let particles_lo = particles.clone();
            let (lo, hi) = rayon::join(
                || Self::tick_split(particles_lo, lo, mid, stepped_time),
                || Self::tick_split(particles, mid, hi, stepped_time)
            );
            lo.into_iter()
                .zip(hi)
                .map(|(a, b)| {
                    a + b
                })
                .collect()
        }
    }

    pub fn get_mass_points(&self) -> Vec<MassPoint> {
        let mut mass_points = Vec::with_capacity(self.particles.len());
        for particle in &*self.particles {
            mass_points.push(MassPoint {
                mass: particle.mass,
                position: particle.position.to_cartesian()
            })
        }
        mass_points
    }
}
