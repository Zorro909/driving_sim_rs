//! Shared native broad phase for a controlled population with car collisions off.
//! Construction follows raw static bodies first, then cars, as in the oracle.
use std::sync::Arc;
use rayon::prelude::*;
use crate::{broadphase::{Aabb,PopulationSpace},car::{Car,Controls,DT},godot_math::F2,world::World};

pub struct NativePopulation {
    pub cars:Vec<Car>,
    world:Arc<World>,
    physics:SharedPhysics,
}
enum PendingShapeUpdate {Car(usize,Aabb),WallCell(Vec<(usize,Aabb)>)}
pub(crate) struct SharedPhysics {
    space:PopulationSpace,
    body_handles:Vec<usize>,
    wall_handles:Vec<usize>,
    cell_order:Vec<(i64,i64)>,
    pending_redraw:Option<Arc<World>>,
    pending_shapes:Vec<PendingShapeUpdate>,
    pending_deletions:Vec<usize>,
}
impl SharedPhysics {
    pub fn resize(&mut self,world:&World,cars:&mut [&mut Car],retained:usize) {
        assert!(retained<=cars.len() && retained<=self.body_handles.len());
        let old=std::mem::take(&mut self.body_handles);
        self.body_handles.extend_from_slice(&old[..retained]);
        for car in cars.iter_mut().skip(retained) {
            let raw=car.update_native_shape(world);
            self.body_handles.push(self.space.add_body(raw));
        }
        // A body joins the pending list once at its first shape edit. The
        // native server drains that list from its head (reverse creation).
        for i in retained..cars.len() {
            let raw=cars[i].update_native_shape(world);
            self.pending_shapes.push(PendingShapeUpdate::Car(self.body_handles[i],raw));
        }
        self.pending_deletions.extend_from_slice(&old[retained..]);
    }
    pub fn new(world:&World,cars:&mut [&mut Car])->Self {
        let walls:Vec<_>=world.track.shapes.iter().map(|s|Aabb{min:F2::from(s.min),max:F2::from(s.max)}).collect();
        let body_handles:Vec<_>=(0..cars.len()).map(|i|if world.track.native_body_order_reverse{cars.len()-1-i}else{i}).collect();
        let mut bodies=vec![Aabb{min:F2::default(),max:F2::default()};cars.len()];
        for (i,car) in cars.iter_mut().enumerate(){bodies[body_handles[i]]=car.update_native_shape(world);}
        let mut space=PopulationSpace::new(&walls,&bodies,world.track.native_tilemap);
        for (i,car) in cars.iter_mut().enumerate().rev(){let raw=car.update_native_shape(world);space.move_body(body_handles[i],raw);}
        if !world.track.native_tilemap {space.finish_setup();}
        Self{space,body_handles,wall_handles:(0..walls.len()).collect(),cell_order:world.track.native_cell_order.clone(),
            pending_redraw:None,pending_shapes:Vec::new(),pending_deletions:Vec::new()}
    }
    pub fn queue_tilemap_redraw(&mut self,old:Arc<World>) {
        assert!(self.pending_redraw.is_none(),"coalesce track edits before queuing native redraw");
        self.pending_redraw=Some(old);
    }
    /// TileMap redraw keeps cell bodies and the BVH allocator, but replaces shapes.
    fn redraw_tilemap(&mut self,old:&World,new:&World) {
        assert!(old.track.native_tilemap && new.track.native_tilemap);
        assert!(old.track.shapes.iter().chain(&new.track.shapes).all(|s|s.tile.is_some()));
        let old_handles=std::mem::replace(&mut self.wall_handles,vec![usize::MAX;new.track.shapes.len()]);
        let mut dirty=self.cell_order.clone();
        for &cell in &new.track.native_tile_order {if !dirty.contains(&cell){dirty.push(cell);}}
        for &cell in dirty.iter().rev() {
            let old_indices:Vec<_>=old.track.shapes.iter().enumerate().filter_map(|(i,s)|(s.tile==Some(cell)).then_some(i)).collect();
            let new_indices:Vec<_>=new.track.shapes.iter().enumerate().filter_map(|(i,s)|(s.tile==Some(cell)).then_some(i)).collect();
            if new_indices.is_empty() {
                if !old_indices.is_empty() {
                    // PhysicsServer.free drains every pending object's shapes first.
                    self.flush_shape_updates();
                    for i in old_indices {self.space.remove_wall(old_handles[i]);}
                }
                continue;
            }
            let mut cache:Vec<_>=old_indices.iter().map(|&i|{
                let s=&old.track.shapes[i];(F2::from(s.min),F2::from(s.max)-F2::from(s.min))
            }).collect();
            // Setting transform, collision layer and mask updates old shapes.
            for _ in 0..3 {
                for (n,&i) in old_indices.iter().enumerate() {
                    let s=&old.track.shapes[i];let old_size=cache[n].1;
                    let margin=((old_size.x+old_size.y) as f64*0.5*0.05) as f32;
                    let d=F2{x:margin,y:margin};
                    let position=F2::from(s.min)-d;let size=(F2::from(s.max)-F2::from(s.min))+d*2.0;
                    cache[n]=(position,size);self.space.move_wall(old_handles[i],Aabb::from_rect(position,size));
                }
            }
            // remove_shape(0) unregisters all following shapes in index order.
            for i in old_indices {self.space.remove_wall(old_handles[i]);}
            self.pending_shapes.push(PendingShapeUpdate::WallCell(new_indices.into_iter().map(|i|{
                let s=&new.track.shapes[i];(i,Aabb{min:F2::from(s.min),max:F2::from(s.max)})
            }).collect()));
        }
        self.cell_order=dirty.into_iter().filter(|c|new.track.native_tile_order.contains(c)).collect();
    }
    fn flush_shape_updates(&mut self) {
        while let Some(update)=self.pending_shapes.pop() {
            match update {
                PendingShapeUpdate::Car(body,raw)=>self.space.move_body(body,raw),
                PendingShapeUpdate::WallCell(shapes)=>for (i,raw) in shapes {self.wall_handles[i]=self.space.add_wall(raw,i);},
            }
        }
    }
    fn process_deferred(&mut self,world:&World) {
        if let Some(old)=self.pending_redraw.take(){self.redraw_tilemap(&old,world);}
        for body in std::mem::take(&mut self.pending_deletions) {
            // QueueFree first exits the tree and unregisters the body. Its
            // destructor then flushes every pending shape before freeing it.
            self.space.remove_body(body);
            self.pending_shapes.retain(|u|!matches!(u,PendingShapeUpdate::Car(id,_) if *id==body));
            self.flush_shape_updates();
        }
        self.flush_shape_updates();
        assert!(self.wall_handles.iter().all(|&h|h!=usize::MAX));
    }
    pub fn advance(&mut self,world:&World,cars:&mut [&mut Car],controls:&[Controls],eliminate_on_wall:bool,drive:bool)->Vec<bool> {
        assert_eq!(controls.len(),cars.len());
        assert_eq!(self.body_handles.len(),cars.len(),"native population size cannot change without updating body allocation state");
        self.process_deferred(world);self.space.update();
        // Wall pairs change only in `update`, so they are fixed for this tick
        // and cars integrate independently. Body moves only touch the bodies
        // tree; they replay in the native list order afterwards.
        for (i,car) in cars.iter_mut().enumerate() {
            let handle=self.body_handles[i];
            let pairs=self.space.wall_pairs(handle);
            let orientation:Vec<_>=pairs.iter().map(|&wall|self.space.wall_first(handle,wall)).collect();
            let shape_indices:Vec<_>=pairs.iter().map(|&wall|self.space.wall_scene_index(wall)).collect();
            car.set_native_pairs(&shape_indices,&orientation,self.space.wall_pair_epochs(handle));
        }
        cars.par_iter_mut().zip(controls).for_each(|(car,c)|car.native_integrate(world,c,DT,eliminate_on_wall,drive));
        for (i,car) in cars.iter().enumerate().rev() {self.space.move_body(self.body_handles[i],car.native_shape_bounds());}
        // Native integration and script state callbacks traverse opposite lists.
        let contacts:Vec<bool>=cars.par_iter_mut().map(|car|car.native_callback(world)).collect();
        for (i,car) in cars.iter().enumerate() {self.space.move_body(self.body_handles[i],car.native_shape_bounds());}
        contacts
    }
}
impl NativePopulation {
    pub fn query_wall_candidates(&self,min:crate::vec2::V2,max:crate::vec2::V2)->Vec<usize> {
        self.physics.space.query_walls(Aabb{min:F2::from(min),max:F2::from(max)})
    }
    pub fn redraw_tilemap(&mut self,world:Arc<World>) {
        self.physics.queue_tilemap_redraw(self.world.clone());self.world=world;
    }
    pub fn extend_cars(&mut self,cars:impl IntoIterator<Item=Car>) {
        let retained=self.cars.len();self.cars.extend(cars);
        self.physics.resize(&self.world,&mut self.cars.iter_mut().collect::<Vec<_>>(),retained);
    }
    pub fn truncate(&mut self,count:usize) {
        assert!(count<=self.cars.len());self.cars.truncate(count);
        self.physics.resize(&self.world,&mut self.cars.iter_mut().collect::<Vec<_>>(),count);
    }
    pub fn new(world:Arc<World>,mut cars:Vec<Car>)->Self {
        let physics=SharedPhysics::new(&world,&mut cars.iter_mut().collect::<Vec<_>>());
        Self{cars,world,physics}
    }
    pub fn step(&mut self,controls:&[Controls],eliminate_on_wall:bool) {
        self.advance(controls,eliminate_on_wall,true);
    }
    pub fn passive_step(&mut self,eliminate_on_wall:bool) {
        self.advance(&vec![Controls::default();self.cars.len()],eliminate_on_wall,false);
    }
    fn advance(&mut self,controls:&[Controls],eliminate_on_wall:bool,drive:bool) {
        self.physics.advance(&self.world,&mut self.cars.iter_mut().collect::<Vec<_>>(),controls,eliminate_on_wall,drive);
    }
}
