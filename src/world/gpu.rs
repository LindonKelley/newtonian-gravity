use std::sync::Arc;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, ImmutableBuffer};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::shader::ShaderModule;
use std::num::NonZeroU16;
use vulkano::{DeviceSize, sync};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::sync::GpuFuture;
use crate::{MassPoint, Particle, Vector};

pub struct GPUWorld {
    device: Arc<Device>,
    queue: Arc<Queue>,
    force_direction_pipeline: Arc<ComputePipeline>,
    acceleration_pipeline: Arc<ComputePipeline>,
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
        let force_direction_shader: Arc<ShaderModule> = force_direction_compute_shader::load(device.clone())
            .expect("failed to create shader");
        let force_direction_pipeline = ComputePipeline::new(
            device.clone(),
            force_direction_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).expect("failed to create compute pipeline");
        let acceleration_compute_shader: Arc<ShaderModule> = acceleration_compute_shader::load(device.clone())
            .unwrap();
        let acceleration_pipeline = ComputePipeline::new(
            device.clone(),
            acceleration_compute_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).unwrap();
        Self {
            device,
            queue,
            force_direction_pipeline,
            acceleration_pipeline,
            particles
        }
    }

    pub fn tick(&mut self, time: f32, steps: NonZeroU16) {
        let stepped_time = time / steps.get() as f32;
        let layout = self.force_direction_pipeline.layout().set_layouts().first().unwrap();
        let particle_length = self.particles.read().unwrap().len();
        let force_direction_buffer_length = particle_length * (particle_length - 1) / 2;
        let (time_buffer, time_buffer_future) = ImmutableBuffer::from_data(stepped_time, BufferUsage::all(), self.queue.clone()).unwrap();
        let force_direction_buffer: Arc<DeviceLocalBuffer<[Vector]>> = DeviceLocalBuffer::array(self.device.clone(), force_direction_buffer_length as DeviceSize, BufferUsage::all(), self.device.active_queue_families()).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.particles.clone()),
                WriteDescriptorSet::buffer(1, time_buffer.clone()),
                WriteDescriptorSet::buffer(2, force_direction_buffer.clone())
            ]
        ).unwrap();
        time_buffer_future
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
        let force_direction_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::MultipleSubmit
            ).unwrap();
            builder
                .bind_pipeline_compute(self.force_direction_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.force_direction_pipeline.layout().clone(),
                    0,
                    set.clone()
                )
                .dispatch([(force_direction_buffer_length / 64 + 1) as u32, 1, 1])
                .unwrap();
            Arc::new(builder.build().unwrap())
        };

        let acceleration_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::MultipleSubmit
            ).unwrap();
            builder
                .bind_pipeline_compute(self.acceleration_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.acceleration_pipeline.layout().clone(),
                    0,
                    set.clone()
                )
                .dispatch([(particle_length / 64 + 1) as u32, 1, 1])
                .unwrap();
            Arc::new(builder.build().unwrap())
        };
        for _ in 0..steps.get() {
            sync::now(self.device.clone())
                .then_execute(self.queue.clone(), force_direction_command_buffer.clone()).unwrap()
                .then_signal_semaphore_and_flush().unwrap()
                .then_execute_same_queue(acceleration_command_buffer.clone()).unwrap()
                .then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();
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

mod force_direction_compute_shader {
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
        if (isinf(f)) {
            force_directions[x].force = 0.0;
        } else {
            force_directions[x].force = f;
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

mod acceleration_compute_shader {
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

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Particles {
    Particle particles[];
};

layout(set = 0, binding = 1) readonly buffer Time {
    float time;
};

layout(set = 0, binding = 2) readonly buffer ForceDirections {
    ForceDirection force_directions[];
};

// if GPU groups are executed in SIMD as I've been reading, then this function
// could have very poor performance due to many group cycles being completely NOP
// because of other members of the group still working on the first for loop
//
// I will look into if this is actually an issue, and fixing it if, so in the
// future, but this will be good enough for now
void main() {
    uint p = gl_GlobalInvocationID.x;
    if (p < particles.length()) {
        float m = particles[p].mass;
        Vector acceleration = Vector(0.0, 0.0);
        for (uint i = 0; i < p; i++) {
            uint x = (particles.length() - i) * (particles.length() - i - 1) / 2 - (p - i);
            float d = force_directions[x].direction + 3.14159265358979323846264338327950288;
            float f = force_directions[x].force;
            if (!isinf(f))
                acceleration = vector_add(acceleration, Vector(d, f / m));
        }
        for (uint j = p+1; j < particles.length(); j++) {
            uint x = (particles.length() - p) * (particles.length() - p - 1) / 2 - (j - p);
            float d = force_directions[x].direction;
            float f = force_directions[x].force;
            if (!isinf(f))
                acceleration = vector_add(acceleration, Vector(d, f / m));
        }
        vector_step(particles[p].velocity, acceleration, time);
        vector_step(particles[p].position, particles[p].velocity, time);
    }
}
"
    }
}
