use nalgebra_glm::{Vec3, Mat4, look_at, perspective};
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;
use std::f32::consts::PI;

mod framebuffer;
mod triangle;
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod camera;
mod texturas;
mod skybox;
mod ray_intersect;
mod material;

use framebuffer::Framebuffer;
use nalgebra_glm::normalize;
use crate::material::Material;
use vertex::Vertex;
use obj::Obj;
use camera::Camera;
use triangle::triangle;
use shaders::{fragment_shader, planeta_gaseoso, lava_shader, cellular_shader, shader_luna, shader_puntas, shader_grupos, shader_agua, bacteria_shader, camo_shader, shader_agujero_negro, shader_variado, anillos_shader}; 
use crate::fragment::Fragment;
use crate::color::Color;
use crate::shaders::vertex_shader;
use noise::{NoiseFn, Simplex};
use fastnoise_lite::FastNoiseLite;
use crate::shaders::vertex_shader_simplex;
use crate::ray_intersect::{Intersect, RayIntersect};
use crate::texturas::TextureManager;
use crate::skybox::Skybox;
use image::open;


pub struct Uniforms {
    model_matrix: Mat4,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    viewport_matrix: Mat4,
    time: u32,
    noise: FastNoiseLite
}

pub struct Uniforms_Simplex {
    model_matrix: Mat4,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    viewport_matrix: Mat4,
    time: u32,
    noise: Simplex,
}

fn crear_ruido_simplex() -> Simplex {
    Simplex::new(100)
}

fn crear_ruido_perlin() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();

    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Perlin));

    noise.set_seed(Some(100)); 
    noise.set_frequency(Some(0.030)); 

    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));
    noise.set_fractal_octaves(Some(9)); 
    noise.set_fractal_lacunarity(Some(1.0)); 
    noise.set_fractal_gain(Some(0.100)); 
    noise.set_fractal_ping_pong_strength(Some(9.0)); 
    noise
}

fn crear_ruido_cellular() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(100)); 
    noise.set_frequency(Some(0.080));  
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::EuclideanSq));  
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Div));  
    noise.set_cellular_jitter(Some(1.0)); 
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));  
    noise.set_fractal_octaves(Some(9));  
    noise.set_fractal_lacunarity(Some(1.0));  
    noise.set_fractal_gain(Some(0.3)); 
    noise 
}

fn crear_ruido_cellular_bacteria() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337)); 
    noise.set_frequency(Some(0.010));  
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::EuclideanSq));  
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Mul));  
    noise.set_cellular_jitter(Some(1.0)); 
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));  
    noise.set_fractal_octaves(Some(3));  
    noise.set_fractal_lacunarity(Some(2.0));  
    noise.set_fractal_gain(Some(1.0)); 
    noise.set_fractal_ping_pong_strength(Some(7.0)); 
    noise 
}

fn crear_ruido_cellular_agujero_negro() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Perlin));
    noise.set_seed(Some(100)); 
    noise.set_frequency(Some(0.030));  
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));  
    noise.set_fractal_octaves(Some(9));  
    noise.set_fractal_lacunarity(Some(1.0));  
    noise.set_fractal_gain(Some(1.0)); 
    noise.set_fractal_weighted_strength(Some(3.0));
    noise.set_fractal_ping_pong_strength(Some(10.0));  
    noise 
}

fn crear_ruido_camo() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
    noise.set_seed(Some(1337)); 
    noise.set_frequency(Some(0.010));  
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::Ridged));  
    noise.set_fractal_octaves(Some(9));  
    noise.set_fractal_lacunarity(Some(5.0));  
    noise.set_fractal_gain(Some(1.0)); 
    noise.set_fractal_weighted_strength(Some(7.0)); 
    noise 
}

fn crear_ruido_variado() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(100)); 
    noise.set_frequency(Some(0.030));  
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));  
    noise.set_fractal_octaves(Some(9));  
    noise.set_fractal_lacunarity(Some(1.0));  
    noise.set_fractal_gain(Some(1.0)); 
    noise.set_fractal_weighted_strength(Some(3.0)); 
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::EuclideanSq));  
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Div));  
    noise.set_cellular_jitter(Some(1.0)); 
    noise 
}

fn crear_ruido_grupos() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337)); 
    noise.set_frequency(Some(0.030));  
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::Hybrid));  
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Sub));  
    noise.set_cellular_jitter(Some(2.0)); 
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));  
    noise.set_fractal_octaves(Some(3));  
    noise.set_fractal_lacunarity(Some(2.0));  
    noise.set_fractal_gain(Some(0.5)); 
    noise.set_fractal_ping_pong_strength(Some(1.0)); 
    noise 
}

fn crear_ruido_cellular_puntas() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337)); 
    noise.set_frequency(Some(0.030));  
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::Manhattan));  
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance));  
    noise.set_cellular_jitter(Some(1.0)); 
    noise 
}

pub fn cast_ray(ray_origin: &Vec3, ray_direction: &Vec3, objects: &[Box<dyn RayIntersect>], color_fondo: &Color) -> Color {
    let mut intersect = Intersect::empty();
    let mut zbuffer = f32::INFINITY;

    for object in objects {
        let tmp = object.ray_intersect(ray_origin, ray_direction);
        if tmp.is_intersecting && tmp.distance < zbuffer {
            zbuffer = tmp.distance;
            intersect = tmp;
        }
    }

    if !intersect.is_intersecting {
        return color_fondo.clone();  
    }

    let mut color = intersect.material.diffuse.clone();
    if let Some(ref textura) = intersect.material.textura {
        color = intersect.material.get_diffuse_color(intersect.u, intersect.v);
    }

    color
}

fn render_skybox(
    framebuffer: &mut Framebuffer, objects: &[Box<dyn RayIntersect>], camera: &Camera, color_fondo: &Color
) {
    let width = framebuffer.width as f32;
    let height = framebuffer.height as f32;
    let aspect_ratio = width / height;
    let fov = PI / 3.0;
    let perspective_scale = (fov * 0.5).tan();

    for y in 0..framebuffer.height {
        for x in 0..framebuffer.width {
            let screen_x = (2.0 * x as f32) / width - 1.0;
            let screen_y = -(2.0 * y as f32) / height + 1.0;

            let screen_x = screen_x * aspect_ratio * perspective_scale;
            let screen_y = screen_y * perspective_scale;

           
            let ray_direction = normalize(&Vec3::new(screen_x, screen_y, -1.0));
            let rotated_direction = camera.base_change(&ray_direction);

           
            let pixel_color = cast_ray(&camera.eye, &rotated_direction, objects, color_fondo);

            framebuffer.set_current_color(pixel_color.to_hex());
            framebuffer.point(x, y, 1.0);
        }
    }
}

fn main() {
    let window_width = 1000;
    let window_height = 800;
    let framebuffer_width = 1000;
    let framebuffer_height = 800;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Planetas",
        window_width,
        window_height,
        WindowOptions::default(),
    )
    .unwrap();

    window.set_position(500, 500);
    window.update();

    let mut manejador_textura = TextureManager::new();
    let imagen_cielo = image::open("assets/sky1.jpg").unwrap().into_rgba8();
    manejador_textura.cargar_textura("cielo", imagen_cielo);
    let textura_cielo = manejador_textura.get_textura("cielo");

    let cielo = Material::new(
        Color::new(255, 255, 255),  
        0.0, 
        [0.0, 0.0],
        textura_cielo.clone(),
        None
    );  

    framebuffer.set_background_color(0x333355);

    let traslaciones = vec![
        Vec3::new(-3.0, 0.0, 0.0), 
        Vec3::new(3.0, 0.0, 0.0),  
        Vec3::new(0.0, -3.0, 0.0), 
        Vec3::new(0.0, 3.0, 0.0),  
        Vec3::new(-3.0, 3.0, 0.0), 
        Vec3::new(3.0, -3.0, 0.0), 
        Vec3::new(-3.0, -3.0, 0.0),
        Vec3::new(3.0, 3.0, 0.0),
        Vec3::new(5.0, 5.0, 0.0),
        Vec3::new(9.0, 9.0, 0.0),   
    ];
    let rotation = Vec3::new(0.0, 0.0, 0.0);
    let rotation_anillos = Vec3::new(PI / 4.0, 0.0, 0.0);
    let scale = 1.0f32;

    let mut camera = Camera::new(
        Vec3::new(50.0, 0.0, 150.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0)
    );

    let obj_sphere = Obj::load("assets/models/sphere.obj").expect("Failed to load sphere.obj");
    let vertex_arrays_sphere = obj_sphere.get_vertex_array();

    let obj_anillos = Obj::load("assets/models/anillos.obj").expect("Failed to load anillos.obj");
    let vertex_arrays_anillos = obj_anillos.get_vertex_array();

    let mut time = 0;
    let mut shader_actual = 1;


    while window.is_open() {
        if window.is_key_down(Key::Escape) {
            break;
        }

        if window.is_key_down(Key::Key1) {
            shader_actual = 1;
        }
        if window.is_key_down(Key::Key2) {
            shader_actual = 2;
        }
        if window.is_key_down(Key::Key3) {
            shader_actual = 3;
        }
        if window.is_key_down(Key::Key4) {
            shader_actual = 4;
        }
        if window.is_key_down(Key::Key5) {
            shader_actual = 5;
        }
        if window.is_key_down(Key::Key6) {
            shader_actual = 6;
        }
        if window.is_key_down(Key::Key7) {
            shader_actual = 7;
        }
        if window.is_key_down(Key::Key8) {
            shader_actual = 8;
        }
        if window.is_key_down(Key::Key9) {
            shader_actual = 9;
        }
        if window.is_key_down(Key::Key0) {
            shader_actual = 0;
        }

        time += 1;

        handle_input(&window, &mut camera);

        framebuffer.clear();

        //let model_matrix = create_model_matrix(translation, scale, rotation);
        let view_matrix = create_view_matrix(camera.eye, camera.center, camera.up);
        let projection_matrix = create_perspective_matrix(window_width as f32, window_height as f32);
        let viewport_matrix = create_viewport_matrix(framebuffer_width as f32, framebuffer_height as f32);


        let noise_perlin = crear_ruido_perlin(); 
        let noise_cellular = crear_ruido_cellular(); 

        let centro = Vec3::new(0.0, 0.0, 0.0);
        let grande = 10000.0;
        let color_fondo = Color::new(135, 206, 235);

        let mut objects: Vec<Box<dyn RayIntersect>> = vec![
            Box::new(Skybox {
                center: centro,
                size: grande,
                materials: [cielo.clone(), cielo.clone(), cielo.clone(), cielo.clone(), cielo.clone(), cielo.clone()],
            }),
        ];
        

        render_skybox(&mut framebuffer, &objects, &camera, &color_fondo);

        let velocidad_orbita = 0.02; 
        let radios_orbitales = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0
        ];
        let velocidades_orbitales = vec![0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];


        let model_matrix_1 = create_model_matrix(Vec3::new(radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).cos(), radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).sin(), 0.0), scale, rotation);
        let model_matrix_anillos = create_model_matrix(Vec3::new(radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).cos(), radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).sin(), 0.0), scale, rotation_anillos);
        let model_matrix_2 = create_model_matrix(Vec3::new(radios_orbitales[0] * (time as f32 * velocidades_orbitales[0]).cos(), radios_orbitales[0] * (time as f32 * velocidades_orbitales[0]).sin(), 0.0), scale, rotation);
        let model_matrix_3 = create_model_matrix(Vec3::new(0.0, 0.0, 0.0), scale, rotation);
        let model_matrix_4 = create_model_matrix(Vec3::new(radios_orbitales[2] * (time as f32 * velocidades_orbitales[2]).cos(), radios_orbitales[2] * (time as f32 * velocidades_orbitales[2]).sin(), 0.0), scale, rotation);
        let model_matrix_5 = create_model_matrix(Vec3::new(radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).cos(), radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).sin(), 0.0), scale, rotation);
        let model_matrix_6 = create_model_matrix(Vec3::new(radios_orbitales[4] * (time as f32 * velocidades_orbitales[4]).cos(), radios_orbitales[4] * (time as f32 * velocidades_orbitales[4]).sin(), 0.0), scale, rotation);
        let model_matrix_7 = create_model_matrix(Vec3::new(radios_orbitales[5] * (time as f32 * velocidades_orbitales[5]).cos(), radios_orbitales[5] * (time as f32 * velocidades_orbitales[5]).sin(), 0.0), scale, rotation);
        let model_matrix_8 = create_model_matrix(Vec3::new(radios_orbitales[6] * (time as f32 * velocidades_orbitales[6]).cos(), radios_orbitales[6] * (time as f32 * velocidades_orbitales[6]).sin(), 0.0), scale, rotation);
        let model_matrix_9 = create_model_matrix(Vec3::new(radios_orbitales[7] * (time as f32 * velocidades_orbitales[7]).cos(), radios_orbitales[7] * (time as f32 * velocidades_orbitales[7]).sin(), 0.0), scale, rotation);
        let model_matrix_10 = create_model_matrix(Vec3::new(radios_orbitales[8] * (time as f32 * velocidades_orbitales[8]).cos(), radios_orbitales[8] * (time as f32 * velocidades_orbitales[8]).sin(), 0.0), scale, rotation);

        let uniforms_gaseoso = Uniforms { 
            model_matrix: model_matrix_1, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_perlin() 
        };

        let uniforms_anillos = Uniforms {
            model_matrix: model_matrix_anillos, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_perlin() 
        };

        let uniforms_cellular_puntas = Uniforms { 
            model_matrix: model_matrix_2, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_cellular_puntas() 
        };

        let uniforms_lava = Uniforms { 
            model_matrix: model_matrix_3, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_perlin() 
        };

        let uniforms_cellular_grupos = Uniforms { 
            model_matrix: model_matrix_4, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_grupos() 
        };

        let uniforms_cellular = Uniforms { 
            model_matrix: model_matrix_5, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_cellular() 
        };

        let uniforms_agua = Uniforms { 
            model_matrix: model_matrix_6, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_perlin() 
        };

        let uniforms_cellular_bacteria = Uniforms { 
            model_matrix: model_matrix_7, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_cellular_bacteria() 
        };

        let uniforms_camo = Uniforms { 
            model_matrix: model_matrix_8, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_camo() 
        };

        let uniforms_cellular_agujero_negro = Uniforms { 
            model_matrix: model_matrix_9, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_cellular_agujero_negro() 
        };

        let uniforms_variado = Uniforms {
            model_matrix: model_matrix_10, 
            view_matrix: view_matrix.clone(), 
            projection_matrix: projection_matrix.clone(), 
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: crear_ruido_variado() 
        };


        framebuffer.set_current_color(0xFFDDDD);

        
            
            
            render_shader(&mut framebuffer, &uniforms_lava, &vertex_arrays_sphere, lava_shader);            
            render_shader(&mut framebuffer, &uniforms_cellular_puntas, &vertex_arrays_sphere, shader_puntas);
            render_shader(&mut framebuffer, &uniforms_gaseoso, &vertex_arrays_sphere, planeta_gaseoso);
            render_shader(&mut framebuffer, &uniforms_anillos, &vertex_arrays_anillos, anillos_shader);
            render_shader(&mut framebuffer, &uniforms_cellular_grupos, &vertex_arrays_sphere, shader_grupos); 
            render_shader(&mut framebuffer, &uniforms_cellular, &vertex_arrays_sphere, cellular_shader);


                   
            let radio_orbita = 2.0; 
            let velocidad_orbita = 0.02; 
            let x_offset = radio_orbita * (time as f32 * velocidad_orbita).cos();
            let z_offset = radio_orbita * (time as f32 * velocidad_orbita).sin();

                
            let translacion_luna = Vec3::new(radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).cos(), radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).sin(), 0.0) + Vec3::new(x_offset, 0.0, z_offset);
            let escala_luna = 0.5; 
            let model_matrix_luna = create_model_matrix(translacion_luna, escala_luna, rotation);

            let luna = Uniforms_Simplex {
                model_matrix: model_matrix_luna,
                view_matrix: view_matrix.clone(),
                projection_matrix: projection_matrix.clone(),
                viewport_matrix: viewport_matrix.clone(),
                time,
                noise: crear_ruido_simplex() 
            };

            render_shader_simplex(&mut framebuffer, &luna, &vertex_arrays_sphere, shader_luna);
            render_shader(&mut framebuffer, &uniforms_agua, &vertex_arrays_sphere, shader_agua);
            render_shader(&mut framebuffer, &uniforms_cellular_bacteria, &vertex_arrays_sphere, bacteria_shader);
            render_shader(&mut framebuffer, &uniforms_camo, &vertex_arrays_sphere, camo_shader);
            render_shader(&mut framebuffer, &uniforms_cellular_agujero_negro, &vertex_arrays_sphere, shader_agujero_negro);
            render_shader(&mut framebuffer, &uniforms_variado, &vertex_arrays_sphere, shader_variado);
           

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

fn render_shader(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    vertex_array: &[Vertex],
    fragment_shader_fn: fn(&Fragment, &Uniforms) -> Color
) {
    let mut transformed_vertices = Vec::with_capacity(vertex_array.len());
    for vertex in vertex_array {
        let transformed = vertex_shader(vertex, uniforms);
        transformed_vertices.push(transformed);
    }

    let mut triangles = Vec::new();
    for i in (0..transformed_vertices.len()).step_by(3) {
        if i + 2 < transformed_vertices.len() {
            triangles.push([
                transformed_vertices[i].clone(),
                transformed_vertices[i + 1].clone(),
                transformed_vertices[i + 2].clone(),
            ]);
        }
    }

    let mut fragments = Vec::new();
    for tri in &triangles {
        fragments.extend(triangle(&tri[0], &tri[1], &tri[2]));
    }

    for fragment in fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;

        if x < framebuffer.width && y < framebuffer.height {
            let shaded_color = fragment_shader_fn(&fragment, &uniforms);
            let color = shaded_color.to_hex();
            framebuffer.set_current_color(color);
            framebuffer.point(x, y, fragment.depth);
        }
    }
}

fn render_shader_simplex(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms_Simplex,
    vertex_array: &[Vertex],
    fragment_shader_fn: fn(&Fragment, &Uniforms_Simplex) -> Color
) {
    let mut transformed_vertices = Vec::with_capacity(vertex_array.len());
    for vertex in vertex_array {
        let transformed = vertex_shader_simplex(vertex, uniforms);
        transformed_vertices.push(transformed);
    }

    let mut triangles = Vec::new();
    for i in (0..transformed_vertices.len()).step_by(3) {
        if i + 2 < transformed_vertices.len() {
            triangles.push([
                transformed_vertices[i].clone(),
                transformed_vertices[i + 1].clone(),
                transformed_vertices[i + 2].clone(),
            ]);
        }
    }

    let mut fragments = Vec::new();
    for tri in &triangles {
        fragments.extend(triangle(&tri[0], &tri[1], &tri[2]));
    }

    for fragment in fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;

        if x < framebuffer.width && y < framebuffer.height {
            let shaded_color = fragment_shader_fn(&fragment, &uniforms);
            let color = shaded_color.to_hex();
            framebuffer.set_current_color(color);
            framebuffer.point(x, y, fragment.depth);
        }
    }
}

fn handle_input(window: &Window, camera: &mut Camera) {
    let movement_speed = 0.1; 
    let rotation_speed = PI / 50.0; 

    let mut movement = Vec3::new(0.0, 0.0, 0.0);

    if window.is_key_down(Key::W) {
        movement += camera.get_forward() * movement_speed;
    }
    if window.is_key_down(Key::S) {
        movement -= camera.get_forward() * movement_speed;
    }

    if window.is_key_down(Key::A) {
        movement -= camera.get_right() * movement_speed;
    }
    if window.is_key_down(Key::D) {
        movement += camera.get_right() * movement_speed;
    }

    if window.is_key_down(Key::Q) {
        movement += camera.get_up() * movement_speed;
    }
    if window.is_key_down(Key::E) {
        movement -= camera.get_up() * movement_speed;
    }

    camera.move_center(movement);


    if window.is_key_down(Key::Left) {
        camera.rotate_y(rotation_speed);
    }
    if window.is_key_down(Key::Right) {
        camera.rotate_y(-rotation_speed);
    }
    if window.is_key_down(Key::Up) {
        camera.rotate_x(rotation_speed);
    }
    if window.is_key_down(Key::Down) {
        camera.rotate_x(-rotation_speed);
    }

    if window.is_key_down(Key::Z) {
        camera.zoom(movement_speed);
    }
    if window.is_key_down(Key::X) {
        camera.zoom(-movement_speed);
    }
}


fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let rotation_matrix_x = Mat4::new(
        1.0,  0.0,    0.0,   0.0,
        0.0,  cos_x, -sin_x, 0.0,
        0.0,  sin_x,  cos_x, 0.0,
        0.0,  0.0,    0.0,   1.0,
    );

    let rotation_matrix_y = Mat4::new(
        cos_y,  0.0,  sin_y, 0.0,
        0.0,    1.0,  0.0,   0.0,
        -sin_y, 0.0,  cos_y, 0.0,
        0.0,    0.0,  0.0,   1.0,
    );

    let rotation_matrix_z = Mat4::new(
        cos_z, -sin_z, 0.0, 0.0,
        sin_z,  cos_z, 0.0, 0.0,
        0.0,    0.0,  1.0, 0.0,
        0.0,    0.0,  0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    let transform_matrix = Mat4::new(
        scale, 0.0,   0.0,   translation.x,
        0.0,   scale, 0.0,   translation.y,
        0.0,   0.0,   scale, translation.z,
        0.0,   0.0,   0.0,   1.0,
    );

    transform_matrix * rotation_matrix
}

fn create_view_matrix(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    look_at(&eye, &center, &up)
}

fn create_perspective_matrix(window_width: f32, window_height: f32) -> Mat4 {
    let fov = 45.0 * PI / 180.0;
    let aspect_ratio = window_width / window_height;
    let near = 0.1;
    let far = 1000.0;

    perspective(fov, aspect_ratio, near, far)
}

fn create_viewport_matrix(width: f32, height: f32) -> Mat4 {
    Mat4::new(
        width / 2.0, 0.0, 0.0, width / 2.0,
        0.0, -height / 2.0, 0.0, height / 2.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
}
