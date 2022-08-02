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
        for particle in &mut self.particles {
            particle.acceleration = Vector::new(0.0, 0.0);
        }
        for i in 0..particles_len {
            for j in i+1..particles_len {
                let (slice_a, slice_b) = self.particles.split_at_mut(j);
                let a = &mut slice_a[i];
                let b = &mut slice_b[0];
                let r_sq = Vector::distance_sq(&a.position, &b.position);
                // Newtons law of universal gravitation: (G * m1 * m2) / r^2
                let f = (6.674_30e-11 * a.mass * b.mass / r_sq) * time;
                {
                    let (x1, y1) = a.position.to_cartesian();
                    let (x2, y2) = b.position.to_cartesian();
                    let d1 = f64::atan2(y2 - y1, x2 - x1);
                    let d2 = f64::atan2(y1 - y2, x1 - x2);
                    // f = ma
                    a.acceleration += Vector::new(f / a.mass, d1);
                    b.acceleration += Vector::new(f / b.mass, d2);
                }
            }
        }
        for particle in &mut self.particles {
            particle.velocity.step(&particle.acceleration, time);
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
    pub velocity: Vector,
    pub acceleration: Vector
}