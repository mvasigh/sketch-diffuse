use nannou::noise::{NoiseFn, Perlin};
use nannou::prelude::*;

const A_INIT: f64 = 1.0;
const B_INIT: f64 = 0.0;
const D_A: f64 = 1.0;
const D_B: f64 = 0.5;
const D_T: f64 = 0.999;
const F: f64 = 0.055;
const K: f64 = 0.062;
const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const MAIN_HUE: f32 = 2.0;
const ACCENT_HUE: f32 = 342.0;

/*
Neighbor indices:
╔═══╦═══╦═══╗
║ 0 ║ 1 ║ 2 ║
╠═══╬═══╬═══╣
║ 3 ║ 4 ║ 5 ║
╠═══╬═══╬═══╣
║ 6 ║ 7 ║ 8 ║
╚═══╩═══╩═══╝
*/
fn diffuse(
    cell: &Cell,
    x: usize,
    y: usize,
    field: &Field,
    scale: f64,
    feed: f64,
    kill: f64,
) -> Cell {
    let top = if y == 0 { HEIGHT - 1 } else { y - 1 };
    let bottom = if y == HEIGHT - 1 { 0 } else { y + 1 };
    let left = if x == 0 { WIDTH - 1 } else { x - 1 };
    let right = if x == WIDTH - 1 { 0 } else { x + 1 };

    let (lap_a, lap_b) = [
        (field.cell_at(left, top), 0.05),
        (field.cell_at(x, top), 0.2),
        (field.cell_at(right, top), 0.05),
        (field.cell_at(left, y), 0.2),
        (field.cell_at(x, y), -1.0),
        (field.cell_at(right, y), 0.2),
        (field.cell_at(left, bottom), 0.05),
        (field.cell_at(x, bottom), 0.2),
        (field.cell_at(right, bottom), 0.05),
    ]
    .iter()
    .fold((0.0, 0.0), |mut acc, (_cell, weight)| {
        acc.0 += _cell.a * weight;
        acc.1 += _cell.b * weight;
        acc
    });
    let a = cell.a;
    let b = cell.b;

    let new_a = a + D_A * lap_a - a * b * b + feed * (1.0 - a) * scale;
    let new_b = b + D_B * lap_b + a * b * b - (kill + feed) * b * scale;

    Cell {
        a: clamp(new_a, 0.0, 1.0),
        b: clamp(new_b, 0.0, 1.0),
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    a: f64,
    b: f64,
}

impl Cell {
    fn new(a: f64, b: f64) -> Cell {
        Cell { a, b }
    }
}

struct Field {
    cells: Vec<Cell>,
    generation: f64,
}

impl Field {
    fn new(width: u32, height: u32) -> Field {
        let cells = (0..(width * height))
            .map(|_i| Cell::new(A_INIT, B_INIT))
            .collect::<Vec<Cell>>();

        Field {
            cells,
            generation: 0.0,
        }
    }

    fn index_at_xy(x: usize, y: usize) -> usize {
        (y * WIDTH) + x
    }

    fn update(&mut self, feed: f64, kill: f64) {
        let mut new_cells = Vec::new();
        for (i, prev_cell) in self.cells.iter().enumerate() {
            let x = i % HEIGHT;
            let y = i / HEIGHT;

            let new_cell = diffuse(prev_cell, x, y, self, D_T, feed, kill);

            new_cells.push(new_cell);
        }
        self.generation += D_T;
        self.cells = new_cells;
    }

    fn cell_at(&self, x: usize, y: usize) -> &Cell {
        &self.cells[Field::index_at_xy(x, y)]
    }

    fn cell_at_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        &mut self.cells[Field::index_at_xy(x, y)]
    }
}

struct Model {
    _window: WindowId,
    texture: wgpu::Texture,
    field: Field,
    noise: Perlin,
}

fn model(app: &App) -> Model {
    let width = WIDTH as u32;
    let height = HEIGHT as u32;
    let _window = app
        .new_window()
        .size(width, height)
        .view(view)
        .build()
        .unwrap();
    let window = app.main_window();

    let texture = wgpu::TextureBuilder::new()
        .size([width, height])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED)
        .build(window.swap_chain_device());

    let mut field = Field::new(width, height);

    let noise = Perlin::default();

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let cell = field.cell_at_mut(x, y);
            let val = noise.get([
                map_range(x, 0, WIDTH, 0.0, 50.0),
                map_range(y, 0, HEIGHT, 0.0, 50.0),
            ]);
            cell.b = if val > 0.1 { 1.0 } else { 0.0 };
            cell.a = if cell.b == 1.0 { 0.0 } else { 1.0 };
        }
    }

    Model {
        _window,
        texture,
        field,
        noise,
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    let z = (_app.elapsed_frames() as f64) * 0.01;
    let feed = map_range(model.noise.get([0.0, 0.0, z]), 0.0, 1.0, 0.054, 0.055);
    let kill = map_range(model.noise.get([0.0, 1.0, z]), 0.0, 1.0, 0.060, 0.063);
    model.field.update(feed, kill);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(WHITE);
    let width = WIDTH as u32;
    let height = HEIGHT as u32;
    let image = nannou::image::ImageBuffer::from_fn(width, height, |x, y| {
        let cell = model.field.cell_at(x as usize, y as usize);
        let clamped = clamp(cell.a - cell.b, 0.0, 1.0);

        let hue = map_range(clamped, 0.0, 1.0, MAIN_HUE, ACCENT_HUE);
        let sat = 1.0 - map_range(clamped, 0.0, 1.0, 0.0, 0.1);
        let lum = 0.5 - map_range(clamped, 0.0, 1.0, 0.0, 0.48);

        let hsl = Hsl::new(hue, sat, lum);
        let rgb = Srgb::from(hsl);
        let r = map_range(rgb.red, 0.0, 1.0, 0, std::u8::MAX);
        let g = map_range(rgb.green, 0.0, 1.0, 0, std::u8::MAX);
        let b = map_range(rgb.blue, 0.0, 1.0, 0, std::u8::MAX);

        nannou::image::Rgba([r, g, b, std::u8::MAX])
    });
    let flat_samples = image.as_flat_samples();
    model.texture.upload_data(
        app.main_window().swap_chain_device(),
        &mut *frame.command_encoder(),
        &flat_samples.as_slice(),
    );

    let draw = app.draw();
    draw.texture(&model.texture);

    draw.to_frame(app, &frame).unwrap();
}

fn main() {
    nannou::app(model).update(update).run();
}
