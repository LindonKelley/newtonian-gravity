use std::ops::{Add, AddAssign};

#[derive(Copy, Clone, Debug)]
pub struct Vector {
    pub direction: f64,
    pub magnitude: f64
}

impl Vector {
    pub fn new(magnitude: f64, direction: f64) -> Self {
        Self { direction, magnitude }
    }

    pub fn from_cartesian(x: f64, y: f64) -> Self {
        Self {
            direction: f64::atan2(y, x),
            magnitude: f64::sqrt(x * x + y * y)
        }
    }

    pub fn scale(&self, scale: f64) -> Self {
        Self { direction: self.direction, magnitude: self.magnitude * scale }
    }

    pub fn step(&mut self, derivative: &Vector, time: f64) {
        let derivative = derivative.scale(time);
        *self = *self + derivative;
    }

    #[allow(unused)]
    pub fn step_components(&mut self, derivative: &Vector, time: f64) {
        let (mut x, mut y) = self.to_cartesian();
        let (dx, dy) = derivative.scale(time).to_cartesian();
        x += dx;
        y += dy;
        self.direction = f64::atan2(y, x);
        self.magnitude = f64::sqrt(x * x + y * y);
    }

    pub fn to_cartesian(&self) -> (f64, f64) {
        let x = self.magnitude * f64::cos(self.direction);
        let y = self.magnitude * f64::sin(self.direction);
        (x, y)
    }

    pub fn distance_sq(&self, other: &Vector) -> f64 {
        let r1 = self.magnitude;
        let r2 = other.magnitude;
        let t1 = self.direction;
        let t2 = other.direction;
        r1 * r1 +
            r2 * r2 -
            2.0 * r1 * r2 * f64::cos(t1 - t2)
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        let r1 = self.magnitude;
        let r2 = rhs.magnitude;
        let t1 = self.direction;
        let t2 = rhs.direction;
        let r3 = f64::sqrt(f64::abs(
            r1 * r1 +
                2.0 * r1 * r2 * f64::cos(t2 - t1) +
                r2 * r2
        ));
        let t3 = t1 + f64::atan2(
            r2 * f64::sin(t2 - t1),
            r1 + r2 * f64::cos(t2 - t1)
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
