use nalgebra_glm::Vec3;
use crate::material::Material;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct Intersect {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub is_intersecting: bool,
    pub material: Material,
    pub u: f32,
    pub v: f32,
}

impl Intersect {
    pub fn new(point: Vec3, normal: Vec3, distance: f32, material: Material, u: f32, v: f32) -> Self {
        Intersect {
            point,
            normal,
            distance,
            is_intersecting: true,
            material,
            u,
            v,
        }
    }

    pub fn empty() -> Self {
        Intersect {
            point: Vec3::zeros(),
            normal: Vec3::zeros(),
            distance: 0.0,
            is_intersecting: false,
            material: Material::black(),
            u: 0.0,
            v: 0.0,
        }
    }
}

pub trait RayIntersect: Sync + Send {
    fn ray_intersect(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> Intersect;
    fn get_uv(&self, point: &Vec3) -> (f32, f32);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}