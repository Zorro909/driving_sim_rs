#[allow(dead_code)]
#[path="managed_vehicle_probe.rs"] mod vehicle;
use altd_sim::{car::{Car,Controls},simulation::NativePopulation,world::World,vec2::V2};
use serde_json::{Value,json};
use std::sync::Arc;
fn main(){
 let args:Vec<_>=std::env::args().collect();let scene:Value=serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
 let log=std::fs::read_to_string(&args[2]).unwrap();
 let states:Vec<Value>=log.lines().filter_map(|l|l.strip_prefix("FIDELITY_FRAME ")).map(|l|serde_json::from_str::<Value>(l).unwrap()).filter(|s|s["phase"]=="before").collect();
 let controls:Vec<Value>=log.lines().filter_map(|l|l.strip_prefix("FIDELITY_CONTROL ")).map(|l|serde_json::from_str(l).unwrap()).collect();
 println!("{}",verify(&scene,&states,&controls));
}
pub fn verify(scene:&Value,states:&[Value],controls:&[Value])->Value {
 let mut world=Arc::new(World::from_scene(scene));
 let count=states.last().unwrap()["tick"].as_u64().unwrap() as usize+1;
 let mut frames=vec![Vec::new();count];let mut outputs=vec![Vec::new();count];
 for state in states {frames[state["tick"].as_u64().unwrap() as usize].push(state);}
 for control in controls {outputs[control["tick"].as_u64().unwrap() as usize].push(control);}
 let cars=frames[0].iter().map(|s|vehicle::restore(&world,s)).collect();let mut sim=NativePopulation::new(world.clone(),cars);
 for _ in 0..8{sim.passive_step(false);}
 let mut errors=Vec::new();let mut exact=0;
 let reset_tick=scene["reset_tick"].as_u64().unwrap_or(90) as usize;
 let empty=Vec::new();let resizes=scene["resize_events"].as_array().unwrap_or(&empty);
 let redraws=scene["redraw_events"].as_array().unwrap_or(&empty);
 let queries=scene["wall_queries"].as_array().unwrap_or(&empty);
 for tick in 0..frames.len()-1 {
  if let Some(event)=redraws.iter().find(|e|resizes.is_empty() && e["tick"].as_u64()==Some(tick as u64)) {
   world=Arc::new(World::from_scene(&event["scene"]));sim.redraw_tilemap(world.clone());
   let position=V2::from(altd_sim::curve::json_vector(&event["spawn"]));let rotation=event["rotation"].as_f64().unwrap();
   for car in &mut sim.cars {car.queue_reset(position,rotation);}
  }
  if resizes.is_empty() && redraws.is_empty() && tick==reset_tick {for car in &mut sim.cars{
   let position=scene.get("reset_position").map(|v|V2::from(altd_sim::curve::json_vector(v))).unwrap_or(V2::new(3840.0,1664.0));
   let rotation=scene["reset_rotation"].as_f64().unwrap_or(std::f32::consts::FRAC_PI_2 as f64);
   car.queue_reset(position,rotation);
  }}
  let controls:Vec<_>=outputs[tick].iter().map(|c|{let c=&c["output"];Controls{acceleration:c[0].as_f64().unwrap(),steering:c[1].as_f64().unwrap(),brake:c[2].as_f64().unwrap(),handbrake:c[3].as_f64().unwrap(),boost:c[4].as_f64().unwrap()}}).collect();
  let waiting=if !redraws.is_empty(){redraws.iter().any(|e|{let t=e["tick"].as_u64().unwrap() as usize;(t..t+8).contains(&tick)})}else if resizes.is_empty(){(reset_tick..reset_tick+8).contains(&tick)}else{resizes.iter().any(|e|{let t=e["tick"].as_u64().unwrap() as usize;(t..t+8).contains(&tick)})};
  if waiting {sim.passive_step(false);}else{sim.step(&controls,false);}
  if let Some(event)=resizes.iter().find(|e|e["tick"].as_u64()==Some((tick+1) as u64)) {
   if let Some(redraw)=redraws.iter().find(|e|e["tick"]==event["tick"]) {
    world=Arc::new(World::from_scene(&redraw["scene"]));sim.redraw_tilemap(world.clone());
   }
   let count=event["count"].as_u64().unwrap() as usize;
   let position=V2::from(altd_sim::curve::json_vector(&event["spawn"]));let rotation=event["rotation"].as_f64().unwrap();
   for car in sim.cars.iter_mut().take(count){car.queue_reset(position,rotation);}
   if count<sim.cars.len(){sim.truncate(count);}else{
    let cars=(sim.cars.len()..count).map(|_|{let mut car=Car::new(&world,position,rotation);car.queue_reset(position,rotation);car}).collect::<Vec<_>>();sim.extend_cars(cars);
   }
  }
  assert_eq!(sim.cars.len(),frames[tick+1].len());
  for (i,(car,s)) in sim.cars.iter().zip(&frames[tick+1]).enumerate(){let bad=vehicle::differences(car,s);if bad.as_object().unwrap().is_empty(){exact+=1;}else{errors.push(json!({"tick":tick+1,"car":i,"differences":bad}));}}
  if let Some(query)=queries.iter().find(|q|q["tick"].as_u64()==Some((tick+1) as u64)) {
   let actual=sim.query_wall_candidates(V2::new(-16384.0,-16384.0),V2::new(16384.0,16384.0));
   if json!(actual)!=query["order"] {errors.push(json!({"tick":tick+1,"wall_order":actual,"expected":query["order"]}));}
  }
 }
 json!({"transitions":states.len()-frames[0].len(),"exact":exact,"errors":errors})
}
