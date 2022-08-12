use std::f32::consts::PI;
use std::sync::Arc;
use rayon::iter::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};
use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::ShaderModule;
use vulkano::DeviceSize;
use vulkano::sync::GpuFuture;
use crate::vector::Vector;

pub struct World {
    pub particles: Vec<Particle>
}

impl World {
    pub fn new() -> Self {
        Self { particles: Vec::new() }
    }

    pub fn tick(&mut self, time: f32) {
        let particles_len = self.particles.len();
        let mut accelerations = vec![Vector::new(0.0, 0.0); particles_len];
        for i in 0..particles_len {
            for j in i+1..particles_len {
                let a = self.particles[i];
                let b = self.particles[j];
                let r_sq = Vector::distance_sq(&a.position, &b.position);
                // Newtons law of universal gravitation: (G * m1 * m2) / r^2
                let mut f = (6.67430e-11 * a.mass * b.mass / r_sq) * time;
                if f.is_infinite() {
                    f = 0.0;
                }
                {
                    let (x1, y1) = a.position.to_cartesian();
                    let (x2, y2) = b.position.to_cartesian();
                    let d1 = f32::atan2(y2 - y1, x2 - x1);
                    let d2 = f32::atan2(y1 - y2, x1 - x2);
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

    pub fn par_tick(&mut self, time: f32) {
        let particles = Arc::new(self.particles.clone());
        let accelerations = Self::tick_split(particles, 0, self.particles.len(), time);
        self.particles.par_iter_mut()
            .zip(accelerations)
            .for_each(|(particle, acceleration)| {
                particle.velocity.step(&acceleration, time);
                particle.position.step(&particle.velocity, time);
            });
    }

    fn tick_split(particles: Arc<Vec<Particle>>, lo: usize, hi: usize, time: f32) -> Vec<Vector> {
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
                    let d1 = f32::atan2(y2 - y1, x2 - x1);
                    let d2 = f32::atan2(y1 - y2, x1 - x2);
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

pub struct GPUWorld {
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipeline: Arc<ComputePipeline>,
    particles: Arc<CpuAccessibleBuffer<[Particle]>>
}

impl GPUWorld {
    pub fn new(particles: Vec<Particle>) -> Self {
        let instance = Instance::new(InstanceCreateInfo::default())
            .expect("failed to create instance");
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .expect("no physical device available");

        let family = physical.queue_families()
            .find(|&q| q.supports_compute())
            .expect("missing compute capabilities");
        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(family)],
                ..Default::default()
            },
        ).expect("failed to create device");
        let queue = queues.next().unwrap();
        let particles = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, particles)
            .expect("failed to create particle buffer");
        // intellij rust plugin failing to auto detect what type this is
        let shader: Arc<ShaderModule> = cs::load(device.clone())
            .expect("failed to create shader");
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).expect("failed to create compute pipeline");
        Self { device, queue, compute_pipeline, particles }
    }

    pub fn tick(&self, time: f32) {
        let layout = self.compute_pipeline.layout().set_layouts().first().unwrap();
        // todo switch over to caching length
        let particle_length = self.particles.read().unwrap().len();
        let force_direction_buffer_length = particle_length * (particle_length - 1) / 2;
        let (time_buffer, time_buffer_future) = ImmutableBuffer::from_data(time, BufferUsage::all(), self.queue.clone())
            .expect("unable to create time buffer");
        // todo for the near future when I push more tick processing to GPU
        //let force_direction_buffer: Arc<DeviceLocalBuffer<[Vector]>> = DeviceLocalBuffer::array(self.device.clone(), force_direction_buffer_length as DeviceSize, BufferUsage::all(), self.device.active_queue_families())
        //    .expect("unable to create force direction buffer");

        //let force_direction_buffer = CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage::all(), false, vec![ForceDirection::default(); force_direction_buffer_length])
        //    .expect("unable to create force direction buffer");

        let force_direction_buffer = unsafe {
            CpuAccessibleBuffer::uninitialized_array(self.device.clone(), force_direction_buffer_length as DeviceSize, BufferUsage::all(), false)
                .expect("unable to create force direction buffer")
        };
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.particles.clone()),
                WriteDescriptorSet::buffer(1, time_buffer),
                WriteDescriptorSet::buffer(2, force_direction_buffer.clone())
            ]
        ).unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();
        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                set
            )
            .dispatch([(force_direction_buffer_length / 64 + 1) as u32, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();

        time_buffer_future
            .then_signal_fence_and_flush()
            .unwrap()
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let force_directions = force_direction_buffer.read().unwrap();
        let mut accelerations = vec![Vector::new(0.0, 0.0); particle_length];
        let mut particles = self.particles.write().unwrap();
        for i in 0..particle_length {
            for j in i + 1..particle_length {
                let index = (particle_length - i) * (particle_length - i - 1) / 2 - (j - i);
                let ForceDirection { force: f, direction: d1 } = force_directions[index];
                let d2 = d1 + PI;
                let a = particles[i];
                let b = particles[j];
                accelerations[i] += Vector::new(f / a.mass, d1);
                accelerations[j] += Vector::new(f / b.mass, d2);
            }
        }
        for (i, particle) in particles.iter_mut().enumerate() {
            particle.velocity.step(&accelerations[i], time);
            particle.position.step(&particle.velocity, time);
        }
    }

    pub fn get_mass_points(&self) -> Vec<MassPoint> {
        let particles = self.particles.read().unwrap();
        let mut mass_points = Vec::with_capacity(particles.len());
        for particle in &*particles {
            mass_points.push(MassPoint {
                mass: particle.mass,
                position: particle.position.to_cartesian()
            })
        }
        mass_points
    }
}

mod cs {
    vulkano_shaders::shader! {
                ty: "compute",
                src: "
#version 450

struct Vector {
    float direction;
    float magnitude;
};

Vector vector_add(Vector self, Vector rhs) {
        float r1 = self.magnitude;
        float r2 = rhs.magnitude;
        float t1 = self.direction;
        float t2 = rhs.direction;
        float r3 = sqrt(abs(
            r1 * r1 +
                2.0 * r1 * r2 * cos(t2 - t1) +
                r2 * r2
        ));
        float t3 = t1 + atan(
            r2 * sin(t2 - t1),
            r1 + r2 * cos(t2 - t1)
        );
        return Vector(t3, r3);
}

Vector vector_scale(Vector self, float scale) {
    return Vector(self.direction, self.magnitude * scale);
}

void vector_step(inout Vector self, Vector derivative, float time) {
    derivative = vector_scale(derivative, time);
    self = vector_add(self, derivative);
}

vec2 vector_to_cartesian(Vector self) {
    float x = self.magnitude * cos(self.direction);
    float y = self.magnitude * sin(self.direction);
    return vec2(x, y);
}

float vector_distance_sq(Vector self, Vector other) {
    float r1 = self.magnitude;
    float r2 = other.magnitude;
    float t1 = self.direction;
    float t2 = other.direction;
    return r1 * r1 +
        r2 * r2 -
        2.0 * r1 * r2 * cos(t1 - t2);
}

struct MassPoint {
    float mass;
    vec2 position;
};

struct Particle {
    float mass;
    Vector position;
    Vector velocity;
};

struct ForceDirection {
    float force;
    float direction;
};

uint leading_zeros(uint n) {
    uint M = 1 << 31;
    for (uint c = 0; c < 32; c++) {
        if ((n & M) == M)
            return c;
        n = n << 1;
    }
    return 32;
}

uint sqrt(uint n) {
    uint MAX_SHIFT = 31;
    uint shift = (MAX_SHIFT - leading_zeros(n)) & ~1u;
    uint bit = 1 << shift;
    uint result = 0;
    while (bit != 0) {
        if (n >= (result + bit)) {
            n = n - (result + bit);
            result = (result >> 1) + bit;
        } else {
            result = result >> 1;
        }
        bit = bit >> 2;
    }
    return result;
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Particles {
    Particle particles[];
};

layout(set = 0, binding = 1) readonly buffer Time {
    float time;
};

layout(set = 0, binding = 2) writeonly buffer ForceDirections {
    ForceDirection force_directions[];
};

void main() {
    uint x = gl_GlobalInvocationID.x;
    if (x < force_directions.length()) {
        uint i = particles.length() - 1 - (1 + sqrt(1 + 8 * x)) / 2;
        uint j = (particles.length() - i) * (particles.length() - i - 1) / 2 + i - x;
        Particle a = particles[i];
        Particle b = particles[j];
        float r_sq = vector_distance_sq(a.position, b.position);
        float f = (6.67430e-11 * a.mass * b.mass / r_sq) * time;
        force_directions[x].force = isinf(f) ? 0.0 : f;
        {
            vec2 p1 = vector_to_cartesian(a.position);
            vec2 p2 = vector_to_cartesian(b.position);
            float d1 = atan(p2.y - p1.y, p2.x - p1.x);
            force_directions[x].direction = d1;
        }
    }
}
"
    }
}

mod cs2 {
    vulkano_shaders::shader! {
                ty: "compute",
                src: "
#version 450

struct Vector {
    float direction;
    float magnitude;
};

struct Particle {
    float mass;
    Vector position;
    Vector velocity;
};

struct ForceDirection {
    float force;
    float direction;
};

layout(set = 0, binding = 0) readonly buffer Particles {
    Particle particles[]; // todo likely error when checking for only length, may have to just pass in the particle length alone like before
};

layout(set = 0, binding = 1) readonly buffer ForceDirections {
    ForceDirection force_directions[];
};

void main() {

}
"
    }
}

mod cs3 {
    vulkano_shaders::shader! {
                ty: "compute",
                src: "
#version 450

void main() {

}
"
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
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
