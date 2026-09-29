//! Godot 4.3 BVH leaf ordering, splitting and incremental optimization.
//! Adapted from core/math/bvh*. See GODOT_LICENSE for the upstream license.
use crate::godot_math::F2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Aabb { pub min: F2, pub max: F2 }
impl Aabb {
    pub fn from_rect(position: F2, size: F2) -> Self { Self { min: position, max: position + size } }
    pub fn size(self) -> F2 { self.max - self.min }
    pub fn grow(self, margin: f32) -> Self {
        let d = F2 { x: margin, y: margin };
        Self { min: self.min - d, max: self.max + d }
    }
    fn merge(self, other: Self) -> Self {
        Self { min: F2 { x: self.min.x.min(other.min.x), y: self.min.y.min(other.min.y) },
               max: F2 { x: self.max.x.max(other.max.x), y: self.max.y.max(other.max.y) } }
    }
    pub fn encloses(self, other: Self) -> bool {
        other.min.x >= self.min.x && other.min.y >= self.min.y && other.max.x <= self.max.x && other.max.y <= self.max.y
    }
    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x && self.min.y <= other.max.y && self.max.y >= other.min.y
    }
    fn proximity(self, other: Self) -> f32 {
        let d = (self.min + self.max) - (other.min + other.max);
        d.x.abs() + d.y.abs()
    }
    fn empty() -> Self { Self { min: F2 { x: f32::MAX, y: f32::MAX }, max: F2 { x: -f32::MAX, y: -f32::MAX } } }
    pub fn roundtrip_rect(self) -> Self { Self::from_rect(self.min, self.size()) }
}

#[derive(Clone)]
struct Node {
    bounds: Aabb,
    parent: Option<usize>,
    children: Option<[usize; 2]>,
    items: Vec<usize>,
    height: usize,
    dirty: bool,
}
impl Node {
    fn leaf(parent: Option<usize>) -> Self {
        Self { bounds: Aabb::empty(), parent, children: None, items: Vec::new(), height: 0, dirty: false }
    }
}

/// One native tree. Item IDs retain insertion order; leaf slots do not.
#[derive(Clone)]
struct Tree {
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    root: usize,
    bounds: Vec<Aabb>,
    owner: Vec<usize>,
}
impl Tree {
    fn insert(&mut self,bounds:Aabb)->usize {
        let id=self.bounds.len();self.bounds.push(bounds);self.owner.push(0);
        let node=self.choose(bounds);let refit=self.add_to_leaf(node,id);
        if refit {self.refit_up(self.nodes[node].parent,true);}
        id
    }
    fn new(bounds: &[Aabb]) -> Self {
        let mut tree = Self { nodes: vec![Node::leaf(None)], free_nodes: Vec::new(), root: 0, bounds: bounds.to_vec(), owner: vec![0; bounds.len()] };
        for id in 0..bounds.len() {
            let node = tree.choose(bounds[id]);
            let refit = tree.add_to_leaf(node, id);
            if refit { tree.refit_up(tree.nodes[node].parent, true); }
        }
        tree
    }
    fn allocate(&mut self, parent: usize) -> usize {
        if let Some(id) = self.free_nodes.pop() { self.nodes[id] = Node::leaf(Some(parent)); id }
        else { let id = self.nodes.len(); self.nodes.push(Node::leaf(Some(parent))); id }
    }
    fn add_to_leaf(&mut self, node: usize, id: usize) -> bool {
        let expanded = self.bounds[id].grow(0.5);
        let n = &mut self.nodes[node];
        let refit = n.items.is_empty() || !n.bounds.encloses(expanded);
        if n.items.is_empty() { n.bounds = expanded; }
        else if refit { n.bounds = n.bounds.merge(expanded); }
        n.items.push(id); self.owner[id] = node; refit
    }
    fn choose(&mut self, bounds: Aabb) -> usize {
        let mut node = self.root;
        loop {
            if let Some([a, b]) = self.nodes[node].children {
                node = if bounds.proximity(self.nodes[a].bounds) < bounds.proximity(self.nodes[b].bounds) { a } else { b };
            } else if self.nodes[node].items.len() < 128 { return node; }
            else { return self.split(node, bounds); }
        }
    }
    fn partition(a: &mut Vec<usize>, b: &mut Vec<usize>, bounds: &[Aabb], axis: usize, center: f32) {
        let mut i = 0;
        while i < a.len() {
            let p = bounds[a[i]].min;
            if (if axis == 0 { p.x } else { p.y }) > center { b.push(a.swap_remove(i)); }
            else { i += 1; }
        }
    }
    fn split(&mut self, node: usize, added: Aabb) -> usize {
        let children = [self.allocate(node), self.allocate(node)];
        let old = std::mem::take(&mut self.nodes[node].items);
        let mut bounds: Vec<Aabb> = old.iter().map(|&i| self.bounds[i]).collect();
        let wildcard = bounds.len(); bounds.push(added);
        let full = self.nodes[node].bounds;
        let size = full.size(); let center = size * 0.5 + full.min;
        // Vector2 chooses X for max-axis ties and Y for min-axis ties.
        let order = if size.x < size.y { [1, 0] } else { [0, 1] };
        let coord = |axis| if axis == 0 { center.x } else { center.y };
        let mut a: Vec<usize> = (0..bounds.len()).collect(); let mut b = Vec::new();
        Self::partition(&mut a, &mut b, &bounds, order[0], coord(order[0]));
        let first_min = a.len().min(b.len());
        if first_min < 32 {
            a.append(&mut b);
            let count = a.iter().filter(|&&i| {
                let p = bounds[i].min; (if order[1] == 0 { p.x } else { p.y }) > coord(order[1])
            }).count();
            let second_min = count.min(a.len() - count);
            let best = if second_min > first_min { 1 } else { 0 };
            if first_min.max(second_min) > 0 { Self::partition(&mut a, &mut b, &bounds, order[best], coord(order[best])); }
        }
        if b.is_empty() { b.push(a.swap_remove(0)); }
        if a.is_empty() { a.push(b.swap_remove(0)); }
        self.nodes[node].children = Some(children);
        let mut target = 0;
        for (child, group) in children.into_iter().zip([a, b]) {
            for index in group {
                if index == wildcard { target = child; }
                else { self.add_to_leaf(child, old[index]); }
            }
        }
        self.refit_up(Some(node), false);
        target
    }
    fn refit_node(&mut self, node: usize) {
        let (bounds, height) = if let Some([a, b]) = self.nodes[node].children {
            (self.nodes[a].bounds.merge(self.nodes[b].bounds), 1 + self.nodes[a].height.max(self.nodes[b].height))
        } else {
            (self.nodes[node].items.iter().fold(Aabb::empty(), |a, &id| a.merge(self.bounds[id])).grow(0.5), 0)
        };
        self.nodes[node].bounds = bounds; self.nodes[node].height = height;
    }
    fn replace_child(&mut self, parent: Option<usize>, old: usize, new: usize) {
        if let Some(parent) = parent {
            let children = self.nodes[parent].children.as_mut().unwrap();
            if children[0] == old { children[0] = new; } else { children[1] = new; }
        } else { self.root = new; }
        self.nodes[new].parent = parent;
    }
    fn balance(&mut self, a: usize) -> usize {
        let Some([b, c]) = self.nodes[a].children else { return a; };
        if self.nodes[a].height == 1 { return a; }
        let delta = self.nodes[c].height as isize - self.nodes[b].height as isize;
        let parent = self.nodes[a].parent;
        if delta > 1 {
            let [f, g] = self.nodes[c].children.unwrap();
            let (keep, moved) = if self.nodes[f].height > self.nodes[g].height { (f, g) } else { (g, f) };
            self.replace_child(parent, a, c);
            self.nodes[c].children = Some([a, keep]); self.nodes[a].parent = Some(c);
            self.nodes[a].children = Some([b, moved]); self.nodes[moved].parent = Some(a);
            self.refit_node(a); self.refit_node(c); c
        } else if delta < -1 {
            let [d, e] = self.nodes[b].children.unwrap();
            let (keep, moved) = if self.nodes[d].height > self.nodes[e].height { (d, e) } else { (e, d) };
            self.replace_child(parent, a, b);
            self.nodes[b].children = Some([keep, a]); self.nodes[a].parent = Some(b);
            self.nodes[a].children = Some([moved, c]); self.nodes[moved].parent = Some(a);
            self.refit_node(a); self.refit_node(b); b
        } else { a }
    }
    fn refit_up(&mut self, mut node: Option<usize>, balance: bool) {
        while let Some(id) = node {
            let id = if balance { self.balance(id) } else { id };
            self.refit_node(id); node = self.nodes[id].parent;
        }
    }
    fn remove(&mut self, id: usize) {
        let node = self.owner[id];
        let refit = !self.nodes[node].bounds.grow(-0.5 - 0.001f32).encloses(self.bounds[id]);
        let index = self.nodes[node].items.iter().position(|&i| i == id).unwrap();
        self.nodes[node].items.swap_remove(index);
        if !self.nodes[node].items.is_empty() { self.nodes[node].dirty |= refit; }
        else if let Some(parent) = self.nodes[node].parent {
            let children = self.nodes[parent].children.unwrap();
            let sibling = if children[0] == node { children[1] } else { children[0] };
            let grandparent = self.nodes[parent].parent;
            self.replace_child(grandparent, parent, sibling);
            self.free_nodes.push(parent); self.free_nodes.push(node);
            self.refit_up(grandparent, false);
        }
    }
    fn refit_dirty(&mut self) {
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if let Some([a, b]) = self.nodes[id].children { stack.push(a); stack.push(b); }
            else if self.nodes[id].dirty { self.nodes[id].dirty = false; self.refit_up(Some(id), false); }
        }
    }
    fn optimize(&mut self, id: usize) {
        self.remove(id);
        let node = self.choose(self.bounds[id]); self.add_to_leaf(node, id);
        self.refit_up(Some(node), true);
    }
    fn move_item(&mut self, id: usize, bounds: Aabb) {
        let node = self.owner[id];
        if self.nodes[node].bounds.encloses(bounds) { self.bounds[id] = bounds; return; }
        self.remove(id); self.bounds[id] = bounds;
        let node = self.choose(bounds);
        if self.add_to_leaf(node, id) { self.refit_up(self.nodes[node].parent, false); }
    }
    fn cull(&self, bounds: Aabb, out: &mut Vec<u32>) {
        out.clear(); let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id]; if !bounds.intersects(node.bounds) { continue; }
            if let Some([a, b]) = node.children { stack.push(a); stack.push(b); }
            else { for &item in &node.items { if bounds.intersects(self.bounds[item]) { out.push(item as u32); } } }
        }
    }
}

/// A fresh static-wall scene followed by one dynamic vehicle.
/// Geometry-only legacy captures cannot supply the prior native tree state.
#[derive(Clone)]
pub(crate) struct BroadPhase {
    walls: Tree,
    cursor: usize,
    pub body_node: Aabb,
    pub initial_pairs: Vec<u32>,
    bodies_first:bool,
}

/// Shared native space for cars whose collision masks exclude other cars.
/// Walls are inserted first. All cars remain in the dynamic tree and optimizer.
#[derive(Clone,Copy)]
enum SpaceItem {Wall(usize),Body(usize)}
pub(crate) struct PopulationSpace {
    walls: Tree,
    bodies: Tree,
    cursor: usize,
    changed: Vec<usize>,
    queued: Vec<bool>,
    pairs: Vec<Vec<usize>>,
    constraints: Vec<Vec<usize>>,
    constraint_epochs:Vec<Vec<u64>>,
    next_constraint_epoch:u64,
    native_refs:Vec<usize>,
    free_refs:Vec<usize>,
    next_ref:usize,
    active_refs:Vec<usize>,
    items:Vec<SpaceItem>,
    wall_ids:Vec<usize>,
    body_ids:Vec<usize>,
    wall_scene_indices:Vec<usize>,
}
impl PopulationSpace {
    pub fn new(walls: &[Aabb], bodies: &[Aabb], bodies_first:bool) -> Self {
        assert!(!bodies.is_empty());
        let count=walls.len()+bodies.len();
        let mut space=Self {
            walls:Tree::new(walls),bodies:Tree::new(bodies),cursor:0,
            changed:Vec::new(),queued:vec![false;count],
            pairs:vec![Vec::new();count],constraints:vec![Vec::new();bodies.len()],
            constraint_epochs:vec![Vec::new();bodies.len()],next_constraint_epoch:1,
            native_refs:(0..count).map(|id|if bodies_first {if id<walls.len(){bodies.len()+id}else{id-walls.len()}}else{id}).collect(),
            free_refs:Vec::new(),next_ref:count,
            active_refs:if bodies_first {(walls.len()..count).chain(0..walls.len()).collect()}else{(0..count).collect()},
            items:(0..walls.len()).map(SpaceItem::Wall).chain((0..bodies.len()).map(SpaceItem::Body)).collect(),
            wall_ids:(0..walls.len()).collect(),body_ids:(walls.len()..count).collect(),
            wall_scene_indices:(0..walls.len()).collect(),
        };
        // BVH.create flushes collision checks immediately. Later dynamic bodies
        // cannot interact with these bodies, so only the static cull matters.
        let mut hits=Vec::new();
        if bodies_first {
            // Deferred TileMap shapes enter the space after all vehicle shapes.
            for (wall,&raw) in walls.iter().enumerate() {
                space.bodies.cull(raw,&mut hits);
                for &body in &hits {
                    let body=body as usize;let id=walls.len()+body;
                    space.pairs[id].push(wall);space.pairs[wall].push(id);space.add_constraint(body,wall);
                }
            }
        } else {for (body,&raw) in bodies.iter().enumerate() {
            space.walls.cull(raw,&mut hits);
            for &wall in &hits {
                let wall=wall as usize;let id=walls.len()+body;
                space.pairs[id].push(wall);space.pairs[wall].push(id);space.add_constraint(body,wall);
            }
        }}
        space
    }
    pub fn finish_setup(&mut self) {
        for id in (0..self.walls.bounds.len()).rev() {
            let raw=self.walls.bounds[id];
            let size=raw.size();let margin=((size.x+size.y) as f64*0.5*0.05) as f32;
            let d=F2{x:margin,y:margin};
            let density=(self.pairs[id].len() as f64*(1.0/9.0)) as f32;
            self.walls.move_item(id,Aabb::from_rect(raw.min-d,size+d*2.0).grow(0.1*(1.0-density.min(1.0))));
            if !self.queued[id]{self.changed.push(id);self.queued[id]=true;}
        }
    }
    fn bounds(&self,id:usize)->Aabb {
        match self.items[id] {SpaceItem::Wall(wall)=>self.walls.bounds[wall],SpaceItem::Body(body)=>self.bodies.bounds[body]}
    }
    pub fn move_body(&mut self,body:usize,raw:Aabb) {
        let id=self.body_ids[body];
        let density=(self.pairs[id].len() as f64*(1.0/9.0)) as f32;
        let expanded=raw.grow(0.1*(1.0-density.min(1.0)));
        let old=self.bodies.bounds[body];let a=old.size();let b=raw.size();
        let within=self.bodies.nodes[self.bodies.owner[body]].bounds.encloses(expanded);
        let threshold=(0.1f32 as f64*2.0*2.0*1.1f32 as f64) as f32;
        if within && old.roundtrip_rect().encloses(raw) && (a.x+a.y)-(b.x+b.y)<threshold{return;}
        self.bodies.move_item(body,expanded);
        if !self.queued[id]{self.changed.push(id);self.queued[id]=true;}
    }
    pub fn update(&mut self) {
        self.walls.refit_dirty();self.bodies.refit_dirty();
        if self.cursor>=self.active_refs.len(){self.cursor=0;}
        if self.active_refs.is_empty(){return;}
        let id=self.active_refs[self.cursor];
        match self.items[id] {SpaceItem::Wall(wall)=>self.walls.optimize(wall),SpaceItem::Body(body)=>self.bodies.optimize(body)}
        self.cursor+=1;
        self.check_pairs();
    }
    fn check_pairs(&mut self) {
        let changed=std::mem::take(&mut self.changed);let mut hits=Vec::new();
        for id in changed {
            let bounds=self.bounds(id).roundtrip_rect();
            let mut i=0;
            while i<self.pairs[id].len() {
                let other=self.pairs[id][i];
                if bounds.intersects(self.bounds(other)){i+=1;continue;}
                self.pairs[id].swap_remove(i);
                let reverse=self.pairs[other].iter().position(|&p|p==id).unwrap();self.pairs[other].swap_remove(reverse);
                let (wall,body)=self.wall_and_body(id,other);
                let index=self.constraints[body].iter().position(|&p|p==wall).unwrap();
                self.constraints[body].remove(index);self.constraint_epochs[body].remove(index);
            }
            let is_wall=matches!(self.items[id],SpaceItem::Wall(_));
            if is_wall{self.bodies.cull(bounds,&mut hits);}else{self.walls.cull(bounds,&mut hits);}
            for &hit in &hits {
                let other=if is_wall{self.body_ids[hit as usize]}else{self.wall_ids[hit as usize]};
                if self.pairs[id].contains(&other){continue;}
                self.pairs[id].push(other);self.pairs[other].push(id);
                let (wall,body)=self.wall_and_body(id,other);
                self.add_constraint(body,wall);
            }
            self.queued[id]=false;
        }
    }
    pub fn wall_pairs(&self,body:usize)->&[usize]{&self.constraints[body]}
    pub fn wall_scene_index(&self,wall:usize)->usize{self.wall_scene_indices[wall]}
    pub fn query_walls(&self,bounds:Aabb)->Vec<usize> {
        let mut hits=Vec::new();self.walls.cull(bounds,&mut hits);
        hits.into_iter().map(|wall|self.wall_scene_indices[wall as usize]).collect()
    }
    fn wall_and_body(&self,a:usize,b:usize)->(usize,usize) {
        match (self.items[a],self.items[b]) {
            (SpaceItem::Wall(wall),SpaceItem::Body(body))|(SpaceItem::Body(body),SpaceItem::Wall(wall))=>(wall,body),
            _=>unreachable!("only wall/body pairs collide"),
        }
    }
    pub fn wall_pair_epochs(&self,body:usize)->&[u64]{&self.constraint_epochs[body]}
    fn add_constraint(&mut self,body:usize,wall:usize) {
        self.constraints[body].push(wall);self.constraint_epochs[body].push(self.next_constraint_epoch);
        self.next_constraint_epoch+=1;
    }
    pub fn wall_first(&self,body:usize,wall:usize)->bool {self.native_refs[self.wall_ids[wall]]<self.native_refs[self.body_ids[body]]}
    pub fn add_body(&mut self,raw:Aabb)->usize {
        let body=self.bodies.insert(raw);let id=self.items.len();
        self.items.push(SpaceItem::Body(body));self.body_ids.push(id);
        let native=self.free_refs.pop().unwrap_or_else(||{let id=self.next_ref;self.next_ref+=1;id});
        self.native_refs.push(native);self.active_refs.push(id);
        self.pairs.push(Vec::new());self.constraints.push(Vec::new());self.constraint_epochs.push(Vec::new());self.queued.push(true);self.changed.push(id);
        self.check_pairs();body
    }
    pub fn remove_body(&mut self,body:usize) {
        let id=self.body_ids[body];
        while !self.pairs[id].is_empty() {
            let wall=self.pairs[id].swap_remove(0);
            let index=self.pairs[wall].iter().position(|&other|other==id).unwrap();self.pairs[wall].swap_remove(index);
        }
        self.constraints[body].clear();
        self.constraint_epochs[body].clear();
        if let Some(index)=self.changed.iter().position(|&other|other==id){self.changed.swap_remove(index);}
        self.queued[id]=false;self.bodies.remove(body);
        let index=self.active_refs.iter().position(|&other|other==id).unwrap();self.active_refs.swap_remove(index);
        self.free_refs.push(self.native_refs[id]);self.check_pairs();
    }
    pub fn add_wall(&mut self,raw:Aabb,scene_index:usize)->usize {
        let wall=self.walls.insert(raw);let id=self.items.len();
        self.items.push(SpaceItem::Wall(wall));self.wall_ids.push(id);self.wall_scene_indices.push(scene_index);
        let native=self.free_refs.pop().unwrap_or_else(||{let next=self.next_ref;self.next_ref+=1;next});
        self.native_refs.push(native);self.active_refs.push(id);self.pairs.push(Vec::new());
        self.queued.push(true);self.changed.push(id);self.check_pairs();wall
    }
    pub fn move_wall(&mut self,wall:usize,raw:Aabb) {
        let id=self.wall_ids[wall];let density=(self.pairs[id].len() as f64*(1.0/9.0)) as f32;
        let expanded=raw.grow(0.1*(1.0-density.min(1.0)));
        let old=self.walls.bounds[wall];let a=old.size();let b=raw.size();
        let within=self.walls.nodes[self.walls.owner[wall]].bounds.encloses(expanded);
        let threshold=(0.1f32 as f64*2.0*2.0*1.1f32 as f64) as f32;
        if within && old.roundtrip_rect().encloses(raw) && (a.x+a.y)-(b.x+b.y)<threshold{return;}
        self.walls.move_item(wall,expanded);
        if !self.queued[id]{self.changed.push(id);self.queued[id]=true;}
    }
    pub fn remove_wall(&mut self,wall:usize) {
        let id=self.wall_ids[wall];
        while !self.pairs[id].is_empty() {
            let other=self.pairs[id].swap_remove(0);
            let reverse=self.pairs[other].iter().position(|&p|p==id).unwrap();self.pairs[other].swap_remove(reverse);
            let (_,body)=self.wall_and_body(id,other);
            let index=self.constraints[body].iter().position(|&p|p==wall).unwrap();
            self.constraints[body].remove(index);self.constraint_epochs[body].remove(index);
        }
        if let Some(index)=self.changed.iter().position(|&other|other==id){self.changed.swap_remove(index);}
        self.queued[id]=false;self.walls.remove(wall);
        let index=self.active_refs.iter().position(|&other|other==id).unwrap();self.active_refs.swap_remove(index);
        self.free_refs.push(self.native_refs[id]);self.check_pairs();
    }
}
impl BroadPhase {
    #[cfg(test)]
    pub fn new(walls: &[Aabb], body: Aabb) -> Self {
        Self::with_layout(walls,body,false)
    }
    pub fn with_layout(walls: &[Aabb], body: Aabb,tilemap:bool) -> Self {
        let mut tree = Tree::new(walls);
        let mut initial_pairs=Vec::new();
        if tilemap {initial_pairs.extend(walls.iter().enumerate().filter(|(_,wall)|body.intersects(**wall)).map(|(id,_)|id as u32));}
        else {tree.cull(body,&mut initial_pairs);}
        // Raw PhysicsServer construction adds a shape before assigning space.
        // set_space inserts it immediately, then the pending shape queue applies
        // a second cache update in reverse body order on the first physics tick.
        if !tilemap {for (id, &raw) in walls.iter().enumerate().rev() {
            let size = raw.size();
            let margin = ((size.x + size.y) as f64 * 0.5 * 0.05) as f32;
            let d = F2 { x: margin, y: margin };
            let cached = Aabb::from_rect(raw.min - d, size + d * 2.0);
            let density=if initial_pairs.contains(&(id as u32)){(1.0f64/9.0) as f32}else{0.0};
            tree.move_item(id, cached.grow(0.1*(1.0-density)));
        }}
        Self { walls: tree, cursor: 0, body_node: body.grow(0.5), initial_pairs,bodies_first:tilemap }
    }
    pub fn update(&mut self, body: Aabb) {
        self.walls.refit_dirty();
        if self.cursor >= self.walls.bounds.len() + 1 { self.cursor = 0; }
        let id=if self.bodies_first {(self.cursor+self.walls.bounds.len())%(self.walls.bounds.len()+1)}else{self.cursor};
        if id < self.walls.bounds.len() { self.walls.optimize(id); }
        else { self.body_node = body.grow(0.5); }
        self.cursor += 1;
    }
    pub fn cull(&self, bounds: Aabb, out: &mut Vec<u32>) { self.walls.cull(bounds, out); }
    pub fn overlaps(&self, id: usize, bounds: Aabb) -> bool { self.walls.bounds[id].roundtrip_rect().intersects(bounds) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_tilemap_deferred_wall_query_order() {
        for data in [include_str!("../tests/fixtures/native_tilemap_a01.json"),include_str!("../tests/fixtures/native_tilemap_b06.json")] {
            let data:serde_json::Value=serde_json::from_str(data).unwrap();
            let walls:Vec<_>=data["walls"].as_array().unwrap().iter().map(|w| {
                let points:Vec<_>=w["world_points"].as_array().unwrap().iter().map(crate::curve::json_vector).collect();
                Aabb{min:F2{x:points.iter().map(|p|p.x).fold(f32::INFINITY,f32::min),y:points.iter().map(|p|p.y).fold(f32::INFINITY,f32::min)},max:F2{x:points.iter().map(|p|p.x).fold(f32::NEG_INFINITY,f32::max),y:points.iter().map(|p|p.y).fold(f32::NEG_INFINITY,f32::max)}}
            }).collect();
            let mut tree=Tree::new(&walls);let mut hits=Vec::new();
            let query=Aabb::from_rect(F2{x:-16384.0,y:-16384.0},F2{x:32768.0,y:32768.0});
            for frame in data["frames"].as_array().unwrap() {
                let tick=frame["tick"].as_u64().unwrap() as usize;
                tree.refit_dirty();
                if tick>3 {tree.optimize(tick-4);}
                tree.cull(query,&mut hits);
                let expected:Vec<_>=frame["walls"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u32).collect();
                assert_eq!(hits,expected,"{} walls, native tick {tick}",walls.len());
            }
        }
    }
    #[test]
    fn original_engine_full_wall_query_order() {
        let data: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/native_broadphase_large80.json")).unwrap();
        let world = crate::world::World::from_scene(&data["scene"]);
        let walls: Vec<_> = world.track.shapes.iter().map(|s| Aabb { min: F2::from(s.min), max: F2::from(s.max) }).collect();
        let body = Aabb::from_rect(F2 { x: 32.0, y: 65.0 }, F2 { x: 62.0, y: 29.0 });
        let query = Aabb::from_rect(F2 { x: -3000.0, y: -3000.0 }, F2 { x: 4000.0, y: 4000.0 });
        let states = data["states"].as_array().unwrap();
        let expected = |s: &serde_json::Value| -> Vec<u32> { s["order"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
        let mut tree = BroadPhase::new(&walls, body); let mut hits = Vec::new();
        // BroadPhaseOracle waits eight native ticks before setting velocity.
        for _ in 0..8 { tree.update(body); }
        tree.cull(query, &mut hits);
        assert_eq!(hits, expected(&states[0]), "initial query order");
        for s in states.iter().skip(1) {
            tree.update(body); tree.cull(query, &mut hits);
            assert_eq!(hits, expected(s), "query order at tick {}", s["tick"]);
        }
    }
}
