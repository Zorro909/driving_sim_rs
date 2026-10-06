//! Original TrackFactory path search, surface assignment and block selection.
//! Each generation takes its own .NET Xoshiro state, independent of reproduction.
use crate::math::profile::MathProfile;
use crate::training::game_random::GameRandom;
use crate::{math::godot_math::F2, track::curve::Curve};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

type Position = [i32; 2];
const NEIGHBORS: [Position; 4] = [[0, -1], [0, 1], [-1, 0], [1, 0]];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomTrackConfig {
    pub length: i32,
    pub(crate) allow_double: bool,
    pub(crate) start: Option<Position>,
    pub(crate) start_direction: Option<usize>,
    pub(crate) surfaces: Option<Vec<usize>>,
    /// Original SurfaceDistribution: Random = 0, Sections = 1.
    pub(crate) distribution: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    pub side: usize,
    pub index: i32,
    #[serde(rename = "type")]
    pub kind: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Block {
    pub atlas: i32,
    pub alternative: i32,
    pub(crate) surface: usize,
    #[serde(rename = "type")]
    pub(crate) kind: usize,
    pub connections: Vec<Connection>,
}

#[derive(Serialize, Debug, PartialEq, Eq)]
struct Lookup {
    surface: usize,
    previous: usize,
    next: usize,
    simple: Vec<usize>,
    all: Vec<usize>,
}

/// The game's tile set as `data/track_catalog.json` holds it. Only what the
/// simulator reads is kept; block candidates are derived (`TrackGenerator`).
#[derive(Deserialize)]
struct Catalog {
    hashcode_seed: u32,
    bounds: [i32; 4],
    blocks: Vec<Block>,
}

fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| serde_json::from_str(include_str!("../data/track_catalog.json")).unwrap())
}

/// .NET 8 HashCode.Combine(int,int), with the process-wide random seed.
/// .NET Foundation, MIT; xxHash32 constants and mixing from Yann Collet.
pub fn combine_hash(seed: u32, a: i32, b: i32) -> i32 {
    let mut h = seed.wrapping_add(374761393).wrapping_add(8);
    for v in [a, b] {
        h = h
            .wrapping_add((v as u32).wrapping_mul(3266489917))
            .rotate_left(17)
            .wrapping_mul(668265263);
    }
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    (h ^ (h >> 16)) as i32
}
pub fn connection_set_hash(seed: u32, connections: &[Connection]) -> i32 {
    let mut hashes: Vec<i32> = connections
        .iter()
        .map(|c| combine_hash(seed, c.side as i32, c.index))
        .collect();
    hashes.sort(); // original orders signed ints, including negative hashes
    hashes.into_iter().fold(0, |h, c| combine_hash(seed, h, c))
}

/// Lookup tables for one original runtime process. Different connection sets
/// with equal integer hashes intentionally share a group in the original game.
pub struct TrackGenerator {
    lookups: Vec<Lookup>,
}
impl TrackGenerator {
    pub fn new(hashcode_seed: u32) -> Self {
        let mut groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, b) in blocks().iter().enumerate() {
            groups
                .entry(connection_set_hash(hashcode_seed, &b.connections))
                .or_default()
                .push(i);
        }
        let mut lookups = Vec::new();
        for surface in 0..3 {
            for previous in 0..4 {
                for next in 0..4 {
                    if previous == next {
                        continue;
                    }
                    let query = |a, b| {
                        connection_set_hash(
                            hashcode_seed,
                            &[
                                Connection {
                                    side: previous,
                                    index: a,
                                    kind: 0,
                                },
                                Connection {
                                    side: next,
                                    index: b,
                                    kind: 0,
                                },
                            ],
                        )
                    };
                    let simple = groups
                        .get(&query(1, 1))
                        .into_iter()
                        .flatten()
                        .copied()
                        .filter(|&i| blocks()[i].surface == surface && blocks()[i].kind == 0)
                        .collect();
                    let mut all = Vec::new();
                    // Type combinations have identical hashes; repeating each pair 25
                    // times cannot add another candidate after the first occurrence.
                    for a in 0..3 {
                        for b in 0..3 {
                            for &i in groups.get(&query(a, b)).into_iter().flatten() {
                                if blocks()[i].surface == surface && !all.contains(&i) {
                                    all.push(i);
                                }
                            }
                        }
                    }
                    lookups.push(Lookup {
                        surface,
                        previous,
                        next,
                        simple,
                        all,
                    });
                }
            }
        }
        Self { lookups }
    }
    pub fn lookup_json(&self) -> Value {
        serde_json::to_value(&self.lookups).unwrap()
    }
    pub fn generate(
        &self,
        config: &mut RandomTrackConfig,
        state: &mut [u64; 4],
    ) -> Result<GeneratedTrack, &'static str> {
        generate_with_lookups(config, state, &self.lookups)
    }
}

/// Catalog and candidate order exported from the unchanged original game DLL.
pub(crate) fn default_hashcode_seed() -> u32 {
    catalog().hashcode_seed
}

pub fn blocks() -> &'static [Block] {
    &catalog().blocks
}

/// Tile grid bounds [x, y, width, height] from the original game catalog.
pub(crate) fn bounds() -> [i32; 4] {
    catalog().bounds
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedTile {
    pub position: Position,
    pub block: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedTrack {
    pub name: String,
    pub config: RandomTrackConfig,
    pub start: Position,
    pub connection: Connection,
    /// Dictionary insertion order is the closed path's traversal order.
    pub tiles: Vec<GeneratedTile>,
}

/// The geometry of each block, in `blocks()` order: wall polygons for the
/// rays and convex collision shapes, relative to the tile centre.
#[derive(Deserialize)]
struct TileResource {
    polygons: Vec<Vec<[f32; 2]>>,
    shapes: Vec<Vec<[f32; 2]>>,
}
/// Every tile's physics material in the game's tile set.
const TILE_FRICTION: f64 = 1.0;
const TILE_BOUNCE: f64 = 0.0;

fn resources() -> &'static [TileResource] {
    static RESOURCES: OnceLock<Vec<TileResource>> = OnceLock::new();
    RESOURCES.get_or_init(|| serde_json::from_str(include_str!("../data/track_resources.json")).unwrap())
}

impl Connection {
    fn position(self, tile: Position) -> F2 {
        let offsets = [128, 384, 640];
        let offset = offsets[self.index as usize];
        let [x, y] = match self.side {
            0 => [offset, 0],
            1 => [offset, 768],
            2 => [0, offset],
            3 => [768, offset],
            _ => unreachable!(),
        };
        F2 {
            x: (x + tile[0] * 768) as f32,
            y: (y + tile[1] * 768) as f32,
        }
    }
    fn normal(self) -> F2 {
        // The original negates Vector2I before converting to Vector2.
        F2 {
            x: -NEIGHBORS[self.side][0] as f32,
            y: -NEIGHBORS[self.side][1] as f32,
        }
    }
}

fn tangent_parameters(base: F2, delta: F2, own: usize, other: usize, math: MathProfile) -> (F2, f32) {
    let smoothness = if matches!(own, 0 | 2 | 100) && matches!(other, 1 | 3 | 4) {
        0.2
    } else {
        0.5
    };
    let angle = match own {
        1 => std::f32::consts::PI / 4.0,
        3 => math.atan2(128.0, 384.0),
        4 => math.atan2(384.0, 128.0),
        _ => return (base, smoothness),
    };
    let cross = base.cross(delta);
    let sign = if cross > 0.0 {
        1.0
    } else if cross < 0.0 {
        -1.0
    } else {
        0.0
    };
    (base.rotated(sign * angle), smoothness)
}

fn vector_json(value: F2) -> Value {
    json!([value.x as f64, value.y as f64])
}

impl GeneratedTrack {
    /// Original TrackUtils.GetPathCurve control points, before native baking.
    pub fn curve_controls(&self, math: MathProfile) -> Value {
        let mut position = self.start;
        let mut connection = self.connection;
        let mut points = Vec::new();
        loop {
            let tile = self
                .tiles
                .iter()
                .find(|tile| tile.position == position)
                .expect("connected generated track");
            points.push((connection.position(position), connection.normal(), connection.kind));
            let outgoing = blocks()[tile.block]
                .connections
                .iter()
                .find(|&&c| c != connection)
                .unwrap();
            position = add(position, NEIGHBORS[outgoing.side]);
            connection = Connection {
                side: outgoing.side ^ 1,
                ..*outgoing
            };
            if position == self.start {
                break;
            }
            assert!(
                points.len() < self.tiles.len(),
                "generated path must return to its start"
            );
        }
        let mut controls = Vec::with_capacity(points.len() + 1);
        for i in 0..points.len() {
            let (position, normal, kind) = points[i];
            let previous = points[(i + points.len() - 1) % points.len()];
            let next = points[(i + 1) % points.len()];
            let incoming_distance = (position - previous.0).length();
            let outgoing_distance = (position - next.0).length();
            let (outgoing, out_smoothness) = tangent_parameters(normal, next.0 - position, kind, next.2, math);
            let (incoming, in_smoothness) = tangent_parameters(-normal, previous.0 - position, kind, previous.2, math);
            controls.push(json!({
                "position":vector_json(position),
                "incoming":vector_json((incoming * incoming_distance) * in_smoothness),
                "outgoing":vector_json((outgoing * outgoing_distance) * out_smoothness),
            }));
        }
        controls.push(controls[0].clone());
        json!({"bake_interval":5.0,"points":controls})
    }

    /// Build the complete scene consumed by World using original TileSet resources.
    /// Vehicle and space settings come from the supplied scene template.
    pub fn to_scene(&self, template: &Value, math: MathProfile) -> Value {
        let controls = self.curve_controls(math);
        let curve = Curve::from_json(&controls, math);
        let mut tiles = Vec::new();
        let mut polygons = Vec::new();
        let mut shapes = Vec::new();
        for tile in &self.tiles {
            let block = &blocks()[tile.block];
            let resource = &resources()[tile.block];
            let origin = F2 {
                x: (tile.position[0] * 768 + 384) as f32,
                y: (tile.position[1] * 768 + 384) as f32,
            };
            tiles.push(json!({"coords":tile.position,"surface":(["asphalt","dirt","ice"][block.surface]),
                "connections":block.connections.iter().map(|c|vector_json(c.position(tile.position))).collect::<Vec<_>>()}));
            for polygon in &resource.polygons {
                polygons.push(
                    json!({"points":polygon.iter().map(|p|vector_json(F2{x:p[0],y:p[1]}+origin)).collect::<Vec<_>>()}),
                );
            }
        }
        let initial_cells: Vec<Position> =
            serde_json::from_str(include_str!("../data/tilemap_initial_cells.json")).unwrap();
        let mut cell_order: Vec<Position> = initial_cells
            .into_iter()
            .filter(|p| self.tiles.iter().any(|t| t.position == *p))
            .collect();
        for tile in &self.tiles {
            if !cell_order.contains(&tile.position) {
                cell_order.push(tile.position);
            }
        }
        // The packed TileMap already contains cells before TrackManager.Init.
        // Surviving coordinates keep their resident order. Dirty cells are
        // processed last-in-first-out; shape order
        // within each cell follows the native polygon decomposition order.
        for position in cell_order.iter().rev() {
            let tile = self.tiles.iter().find(|t| t.position == *position).unwrap();
            let resource = &resources()[tile.block];
            let origin = F2 {
                x: (tile.position[0] * 768 + 384) as f32,
                y: (tile.position[1] * 768 + 384) as f32,
            };
            for shape in &resource.shapes {
                shapes.push(
                    json!({"tile":tile.position,"origin":vector_json(origin),"local_points":shape,
                    "points":shape.iter().map(|p|vector_json(F2{x:p[0],y:p[1]}+origin)).collect::<Vec<_>>(),
                    "friction":TILE_FRICTION,"bounce":TILE_BOUNCE,"wall_first":false}),
                );
            }
        }
        let mut walls = Vec::new();
        if let Some(tree) = crate::track::bsp::Node::from_polygons(&polygons) {
            tree.collect(&mut walls);
        }
        let direction = curve.direction(0.0);
        let rotation = math.atan2(direction.y, direction.x) + std::f32::consts::PI / 2.0;
        let mut scene = template.clone();
        scene["reset_position"] = vector_json(curve.position(0.0));
        scene["reset_rotation"] = json!(rotation as f64);
        scene["track"] = json!({"name":self.name,"tile_size":[768,768],"tile_map_position":[0,0],
            "native_creation":"tilemap","native_broadphase":true,"raycaster_present":true,
            "native_cell_order":cell_order,
            "tiles":tiles,"polygons":polygons,"physics_shapes":shapes,
            "walls":walls.iter().map(|w|json!([vector_json(w.start.into()),vector_json(w.end.into())])).collect::<Vec<_>>(),
            "curve":controls,"path":curve.points.iter().copied().map(vector_json).collect::<Vec<_>>(),
            "path_forward":curve.forwards.iter().copied().map(vector_json).collect::<Vec<_>>()});
        scene
    }
}

/// Advances the supplied independent track RNG state, including on failure.
/// A successful direct attempt pins `config.start`; fallback changes a new config.
pub fn generate(config: &mut RandomTrackConfig, state: &mut [u64; 4]) -> Result<GeneratedTrack, &'static str> {
    static DEFAULT: OnceLock<TrackGenerator> = OnceLock::new();
    DEFAULT
        .get_or_init(|| TrackGenerator::new(default_hashcode_seed()))
        .generate(config, state)
}
fn generate_with_lookups(
    config: &mut RandomTrackConfig,
    state: &mut [u64; 4],
    lookups: &[Lookup],
) -> Result<GeneratedTrack, &'static str> {
    let mut random = GameRandom::new(*state, 0);
    let result = generate_internal(config, &mut random, &mut 0, lookups);
    *state = random.decision_state;
    result
}

fn add(a: Position, b: Position) -> Position {
    [a[0] + b[0], a[1] + b[1]]
}
fn side(from: Position, to: Position) -> usize {
    NEIGHBORS
        .iter()
        .position(|&d| add(from, d) == to)
        .expect("cardinal path neighbor")
}
fn neighbors(position: Position) -> impl Iterator<Item = Position> {
    NEIGHBORS.into_iter().map(move |delta| add(position, delta))
}
fn in_bounds(position: Position) -> bool {
    let [x, y, width, height] = catalog().bounds;
    position[0] >= x && position[1] >= y && position[0] < x + width && position[1] < y + height
}
fn shuffle<T>(items: &mut [T], random: &mut GameRandom) {
    for i in (1..items.len()).rev() {
        items.swap(i, random.randrange(i + 1));
    }
}

fn generate_internal(
    config: &mut RandomTrackConfig,
    random: &mut GameRandom,
    attempts: &mut usize,
    lookups: &[Lookup],
) -> Result<GeneratedTrack, &'static str> {
    for _ in 0..50 {
        if *attempts >= 200 {
            break;
        }
        *attempts += 1;
        let start = config.start.unwrap_or_else(|| {
            let [x, y, width, height] = catalog().bounds;
            [
                x + random.randrange(width as usize) as i32,
                y + random.randrange(height as usize) as i32,
            ]
        });
        let mut path = vec![start];
        let mut visited = HashSet::from([start]);
        if !build_path(&mut path, &mut visited, start, config.length, random, &mut 0) {
            continue;
        }
        let surfaces = assign_surfaces(&path, config, random);
        let Some(selected) = assign_blocks(&path, &surfaces, config.allow_double, random, lookups) else {
            continue;
        };
        let direction = side(path[0], path[1]);
        let connection = first_connection(selected[0], direction);
        if !finish_compatible(connection) || !complete(&path, &selected) {
            continue;
        }
        config.start.get_or_insert(start);
        config.start_direction.get_or_insert(direction);
        return Ok(GeneratedTrack {
            name: format!(
                "Random ({}, {})",
                if config.allow_double { "All" } else { "Simple" },
                config.length
            ),
            config: config.clone(),
            start,
            connection,
            tiles: path
                .into_iter()
                .zip(selected)
                .map(|(position, block)| GeneratedTile { position, block })
                .collect(),
        });
    }
    if config.length > 4 && *attempts < 200 {
        let mut shorter = config.clone();
        shorter.length -= 2;
        return generate_internal(&mut shorter, random, attempts, lookups);
    }
    Err("Failed to generate a random track")
}

fn build_path(
    path: &mut Vec<Position>,
    visited: &mut HashSet<Position>,
    start: Position,
    length: i32,
    random: &mut GameRandom,
    calls: &mut usize,
) -> bool {
    *calls += 1;
    if *calls > 5000 {
        return false;
    }
    let current = *path.last().unwrap();
    if path.len() as i32 == length {
        return (start[0] - current[0]).abs() + (start[1] - current[1]).abs() == 1;
    }
    let mut next: Vec<_> = neighbors(current)
        .filter(|p| in_bounds(*p) && !visited.contains(p))
        .collect();
    shuffle(&mut next, random);
    for position in next {
        path.push(position);
        visited.insert(position);
        if build_path(path, visited, start, length, random, calls) {
            return true;
        }
        path.pop();
        visited.remove(&position);
    }
    false
}

fn assign_surfaces(path: &[Position], config: &RandomTrackConfig, random: &mut GameRandom) -> Vec<usize> {
    let surfaces = config
        .surfaces
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&[0, 1, 2]);
    if surfaces.len() == 1 || config.distribution == 0 {
        // Random.Next(1) still consumes a draw in the original runtime.
        return path
            .iter()
            .map(|_| surfaces[random.randrange(surfaces.len())])
            .collect();
    }
    let mut result = Vec::with_capacity(path.len());
    for (i, &surface) in surfaces.iter().enumerate() {
        result.extend(std::iter::repeat_n(
            surface,
            path.len() / surfaces.len() + usize::from(i < path.len() % surfaces.len()),
        ));
    }
    result
}

fn candidates(lookups: &[Lookup], previous: usize, next: usize, surface: usize, all: bool) -> Vec<usize> {
    let lookup = lookups
        .iter()
        .find(|row| row.previous == previous && row.next == next && row.surface == surface);
    lookup.map_or_else(Vec::new, |row| if all { row.all.clone() } else { row.simple.clone() })
}
fn compatible(a: usize, b: usize, touching_side: usize) -> bool {
    blocks()[a]
        .connections
        .iter()
        .filter(|c| c.side == touching_side)
        .all(|c| blocks()[b].connections.contains(&Connection { side: c.side ^ 1, ..*c }))
}
fn compatible_both(a: usize, b: usize, touching_side: usize) -> bool {
    compatible(a, b, touching_side) && compatible(b, a, touching_side ^ 1)
}
fn empty_compatible(block: usize, touching_side: usize) -> bool {
    blocks()[block].connections.iter().all(|c| c.side != touching_side)
}
fn empty_neighbors_compatible(position: Position, block: usize, path: &[Position]) -> bool {
    neighbors(position)
        .filter(|p| in_bounds(*p) && !path.contains(p))
        .all(|p| empty_compatible(block, side(position, p)))
}
fn first_connection(block: usize, side: usize) -> Connection {
    blocks()[block]
        .connections
        .iter()
        .find(|c| c.side == side)
        .copied()
        .unwrap_or(Connection {
            side: 0,
            index: 0,
            kind: 0,
        })
}
fn finish_compatible(connection: Connection) -> bool {
    connection.kind == 0 || connection.kind == 100
}

fn assign_blocks(
    path: &[Position],
    surfaces: &[usize],
    all: bool,
    random: &mut GameRandom,
    lookups: &[Lookup],
) -> Option<Vec<usize>> {
    let mut selected = vec![0; path.len()];
    if all {
        return assign_all(path, surfaces, &mut selected, 0, random, &mut 0, lookups).then_some(selected);
    }
    for i in 0..path.len() {
        let previous = side(path[i], path[(i + path.len() - 1) % path.len()]);
        let next = side(path[i], path[(i + 1) % path.len()]);
        let choices = candidates(lookups, previous, next, surfaces[i], false);
        if choices.is_empty() {
            return None;
        }
        let block = choices[random.randrange(choices.len())];
        if !empty_neighbors_compatible(path[i], block, path) {
            return None;
        }
        selected[i] = block;
    }
    (0..path.len())
        .all(|i| {
            let next = (i + 1) % path.len();
            compatible_both(selected[i], selected[next], side(path[i], path[next]))
        })
        .then_some(selected)
}

fn assign_all(
    path: &[Position],
    surfaces: &[usize],
    selected: &mut [usize],
    index: usize,
    random: &mut GameRandom,
    calls: &mut usize,
    lookups: &[Lookup],
) -> bool {
    *calls += 1;
    if *calls > 5000 {
        return false;
    }
    let count = path.len();
    if index == count {
        return compatible_both(selected[count - 1], selected[0], side(path[count - 1], path[0]));
    }
    let previous = side(path[index], path[(index + count - 1) % count]);
    let next = side(path[index], path[(index + 1) % count]);
    let mut choices = candidates(lookups, previous, next, surfaces[index], true);
    shuffle(&mut choices, random);
    for block in choices {
        if *calls > 5000 {
            return false;
        }
        if index > 0 && !compatible_both(selected[index - 1], block, previous ^ 1) {
            continue;
        }
        if index == 0 && !finish_compatible(first_connection(block, next)) {
            continue;
        }
        if index == count - 1 && !compatible_both(block, selected[0], next) {
            continue;
        }
        if !empty_neighbors_compatible(path[index], block, path) {
            continue;
        }
        selected[index] = block;
        if assign_all(path, surfaces, selected, index + 1, random, calls, lookups) {
            return true;
        }
    }
    false
}

fn complete(path: &[Position], selected: &[usize]) -> bool {
    path.len() >= 4
        && path.iter().enumerate().all(|(i, &position)| {
            neighbors(position)
                .enumerate()
                .all(|(side, neighbor)| match path.iter().position(|p| *p == neighbor) {
                    Some(j) => compatible(selected[i], selected[j], side),
                    None => empty_compatible(selected[i], side),
                })
        })
}

/// The game's track TileMap sits at this global position; physics shapes
/// follow it, while ray walls and the path are absolute.
pub(crate) const GAME_TILE_MAP_POSITION: [i32; 2] = [3, 0];

fn saved_coords(value: &Value, what: &str) -> Result<Position, String> {
    let text = value.as_str().ok_or_else(|| format!("{what} must be a string"))?;
    let values = text
        .strip_prefix("Vector2i(")
        .and_then(|value| value.strip_suffix(')'))
        .map(|inner| {
            inner
                .split(',')
                .map(|v| v.trim().parse::<i32>())
                .collect::<Result<Vec<_>, _>>()
        })
        .and_then(Result::ok)
        .filter(|values| values.len() == 2)
        .ok_or_else(|| format!("{what} {text:?} is not Vector2i(x, y)"))?;
    Ok([values[0], values[1]])
}

fn saved_int(value: &Value, what: &str) -> Result<i64, String> {
    value.as_i64().ok_or_else(|| format!("{what} must be an integer"))
}

impl GeneratedTrack {
    /// A track the game saved: the JSON inside a gzip `.track` file. Tiles,
    /// the start tile and its connection are all a scene needs. The path must
    /// close; it is checked here, as `curve_controls` assumes it.
    pub(crate) fn from_saved(saved: &Value, name: &str) -> Result<GeneratedTrack, String> {
        let entries = saved["Tiles"]
            .as_object()
            .ok_or("the file has no Tiles; it is not a saved track")?;
        if entries.is_empty() {
            return Err("the track has no tiles".into());
        }
        let mut tiles = Vec::with_capacity(entries.len());
        for (key, tile) in entries {
            let position = saved_coords(&Value::String(key.clone()), "tile position")?;
            let id = &tile["Block"]["TileSetId"];
            let atlas = saved_int(&id["AtlasId"], "AtlasId")?;
            let alternative = saved_int(&id["AlternativeId"], "AlternativeId")?;
            let block = blocks()
                .iter()
                .position(|b| b.atlas as i64 == atlas && b.alternative as i64 == alternative)
                .ok_or_else(|| {
                    format!("tile {key} uses an unknown block (atlas {atlas}, alternative {alternative})")
                })?;
            tiles.push(GeneratedTile { position, block });
        }
        let start = saved_coords(&saved["StartTileCoords"], "StartTileCoords")?;
        let c = &saved["StartTileConnection"];
        let side = saved_int(&c["Side"], "StartTileConnection.Side")?;
        let index = saved_int(&c["Index"], "StartTileConnection.Index")?;
        let kind = saved_int(&c["Type"], "StartTileConnection.Type")?;
        if !(0..4).contains(&side) || !(0..3).contains(&index) || kind < 0 {
            return Err("the start connection is out of range".into());
        }
        let connection = Connection {
            side: side as usize,
            index: index as i32,
            kind: kind as usize,
        };
        // Walk the path as curve_controls does.
        let (mut position, mut incoming) = (start, connection);
        for _ in 0..tiles.len() {
            let tile = tiles
                .iter()
                .find(|t| t.position == position)
                .ok_or_else(|| format!("the path leaves the track at tile ({}, {})", position[0], position[1]))?;
            let connections = &blocks()[tile.block].connections;
            if !connections.contains(&incoming) {
                return Err(format!("the path is broken at tile ({}, {})", position[0], position[1]));
            }
            let outgoing = connections
                .iter()
                .find(|&&c| c != incoming)
                .ok_or_else(|| format!("tile ({}, {}) has no exit", position[0], position[1]))?;
            position = add(position, NEIGHBORS[outgoing.side]);
            incoming = Connection {
                side: outgoing.side ^ 1,
                ..*outgoing
            };
            if position == start {
                return Ok(GeneratedTrack {
                    name: name.to_owned(),
                    config: RandomTrackConfig {
                        length: 0,
                        allow_double: true,
                        start: None,
                        start_direction: None,
                        surfaces: None,
                        distribution: 0,
                    },
                    start,
                    connection,
                    tiles,
                });
            }
        }
        Err("the track's path does not return to its start".into())
    }
}

/// Moves a scene's TileMap to `position`, as the game places it. Explicit
/// shape origins move with it.
pub fn place_tile_map(scene: &mut Value, position: [i32; 2]) {
    let track = &mut scene["track"];
    let previous = crate::track::world::vector(&track["tile_map_position"]);
    track["tile_map_position"] = json!(position);
    if let Some(shapes) = track["physics_shapes"].as_array_mut() {
        for shape in shapes {
            if let Some(origin) = shape.get("origin").map(crate::track::world::vector) {
                shape["origin"] = json!([
                    origin.x - previous.x + position[0] as f64,
                    origin.y - previous.y + position[1] as f64
                ]);
            }
        }
    }
}

/// A saved game track as a training scene for independent cars, using the
/// vehicle and physics of `template`.
pub fn saved_track_scene(saved: &Value, name: &str, template: &Value, math: MathProfile) -> Result<Value, String> {
    let mut scene = GeneratedTrack::from_saved(saved, name)?.to_scene(template, math);
    place_tile_map(&mut scene, GAME_TILE_MAP_POSITION);
    scene["track"]["native_broadphase"] = Value::Bool(false);
    Ok(scene)
}

#[cfg(test)]
mod saved_tests {
    use super::*;

    /// A generated track in the game's saved `.track` layout.
    fn saved(track: &GeneratedTrack) -> Value {
        let tiles: serde_json::Map<String, Value> = track
            .tiles
            .iter()
            .map(|t| {
                let key = format!("Vector2i({}, {})", t.position[0], t.position[1]);
                let b = &blocks()[t.block];
                (
                    key.clone(),
                    json!({"Block":{"TileSetId":{"AtlasId":b.atlas,"AlternativeId":b.alternative}},"Coords":key}),
                )
            })
            .collect();
        json!({"Tiles":tiles,"StartTileCoords":format!("Vector2i({}, {})", track.start[0], track.start[1]),
            "StartTileConnection":{"Side":track.connection.side,"Index":track.connection.index,"Type":track.connection.kind}})
    }

    /// Prints a generated track in the saved layout: the web app's test
    /// fixture (`cargo test --lib print_saved_fixture -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn print_saved_fixture() {
        let mut config = RandomTrackConfig {
            length: 12,
            allow_double: false,
            start: None,
            start_direction: None,
            surfaces: Some(vec![0, 1]),
            distribution: 1,
        };
        let track = generate(&mut config, &mut [7, 3, 5, 11]).unwrap();
        println!("{}", saved(&track));
    }

    #[test]
    fn saved_tracks_round_trip_and_reject_open_paths() {
        let mut config = RandomTrackConfig {
            length: 16,
            allow_double: true,
            start: None,
            start_direction: None,
            surfaces: Some(vec![0, 1]),
            distribution: 1,
        };
        let track = generate(&mut config, &mut [1, 2, 3, 4]).unwrap();
        let file = saved(&track);
        let parsed = GeneratedTrack::from_saved(&file, "test").unwrap();
        assert_eq!(
            (parsed.start, parsed.connection, &parsed.tiles),
            (track.start, track.connection, &track.tiles)
        );
        let scene = saved_track_scene(&file, "test", &json!({"vehicle":{},"physics":{}}), MathProfile::Proton).unwrap();
        assert_eq!(scene["track"]["tile_map_position"], json!(GAME_TILE_MAP_POSITION));
        assert_eq!(scene["track"]["native_broadphase"], json!(false));
        let shape = &scene["track"]["physics_shapes"][0];
        let tile = shape["tile"].as_array().unwrap();
        assert_eq!(
            shape["origin"],
            json!([
                tile[0].as_f64().unwrap() * 768.0 + 387.0,
                tile[1].as_f64().unwrap() * 768.0 + 384.0
            ])
        );

        let mut open = file.clone();
        let removed = format!(
            "Vector2i({}, {})",
            track.tiles[3].position[0], track.tiles[3].position[1]
        );
        open["Tiles"].as_object_mut().unwrap().remove(&removed);
        assert!(GeneratedTrack::from_saved(&open, "open")
            .unwrap_err()
            .contains("leaves the track"));
        let mut unknown = file.clone();
        unknown["Tiles"][&removed]["Block"]["TileSetId"]["AtlasId"] = json!(999);
        assert!(GeneratedTrack::from_saved(&unknown, "unknown")
            .unwrap_err()
            .contains("unknown block"));
        assert!(GeneratedTrack::from_saved(&json!({"Id": 1}), "other").is_err());
    }
}
