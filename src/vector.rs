use std::ops::{Add, AddAssign};
use bytemuck::{Pod, Zeroable};

#[derive(Default, Copy, Clone, Debug, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Vector {
    pub direction: f32,
    pub magnitude: f32
}

impl Vector {
    pub fn new(magnitude: f32, direction: f32) -> Self {
        Self { direction, magnitude }
    }

    pub fn from_cartesian(x: f32, y: f32) -> Self {
        Self {
            direction: f32::atan2(y, x),
            magnitude: f32::sqrt(x * x + y * y)
        }
    }

    pub fn scale(&self, scale: f32) -> Self {
        Self { direction: self.direction, magnitude: self.magnitude * scale }
    }

    pub fn step(&mut self, derivative: &Vector, time: f32) {
        let derivative = derivative.scale(time);
        *self = *self + derivative;
    }

    pub fn to_cartesian(&self) -> (f32, f32) {
        let x = self.magnitude * f32::cos(self.direction);
        let y = self.magnitude * f32::sin(self.direction);
        (x, y)
    }

    pub fn distance_sq(&self, other: &Vector) -> f32 {
        let r1 = self.magnitude;
        let r2 = other.magnitude;
        let t1 = self.direction;
        let t2 = other.direction;
        r1 * r1 +
            r2 * r2 -
            2.0 * r1 * r2 * f32::cos(t1 - t2)
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        let r1 = self.magnitude;
        let r2 = rhs.magnitude;
        let t1 = self.direction;
        let t2 = rhs.direction;
        let r3 = f32::sqrt(f32::abs(
            r1 * r1 +
                2.0 * r1 * r2 * f32::cos(t2 - t1) +
                r2 * r2
        ));
        let t3 = t1 + f32::atan2(
            r2 * f32::sin(t2 - t1),
            r1 + r2 * f32::cos(t2 - t1)
        );
        Self {
            direction: t3,
            magnitude: r3
        }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
