use bytemuck::{Pod, Zeroable};
use crate::vector::Vector;

pub mod cpu;
pub mod par;
pub mod gpu;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct MassPoint {
    pub mass: f32,
    pub position: (f32, f32)
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Particle {
    pub mass: f32,
    pub position: Vector,
    pub velocity: Vector
}

#[derive(Default, Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
struct ForceDirection {
    force: f32,
    direction: f32
}
