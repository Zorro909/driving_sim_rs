use altd_sim::{car::{Car,Controls,DT},godot_math::F2,vec2::V2,world::World,native_math};
use serde_json::{Value,json};
fn vec(v:&Value)->V2 {
 if let Some(bits)=v["bits"].as_array(){return V2::new(f32::from_bits(bits[0].as_u64().unwrap() as u32) as f64,f32::from_bits(bits[1].as_u64().unwrap() as u32) as f64);}
 V2::new(v["x"].as_f64().unwrap(),v["y"].as_f64().unwrap())
}
pub fn restore(w:&World,s:&Value)->Car {
 let t=&s["physical"];let x=F2::from(vec(&t["x"]));let y=F2::from(vec(&t["y"]));
 let mut c=Car::new(w,vec(&t["origin"]),native_math::atan2(x.y,x.x) as f64);
 c.active=s["active"].as_bool().unwrap_or(true);
 c.set_body_basis(x,y);c.velocity=vec(&s["velocity"]);c.angular_velocity=s["angular"].as_f64().unwrap();c.boost_energy=s["boost"].as_f64().unwrap();
 for (wheel,sw) in c.wheels.iter_mut().zip(s["wheels"].as_array().unwrap()) {
  wheel.angle_deg=sw["_currentRotationDegrees"].as_f64().unwrap() as f32 as f64;
  wheel.previous_position=Some(vec(&sw["_lastPosition"]));
 }
 c
}
pub fn differences(c:&Car,s:&Value)->Value {
 let t=&s["physical"];let mut bad=serde_json::Map::new();
 if let Some(active)=s["active"].as_bool(){if c.active!=active{bad.insert("active".into(),json!([c.active,active]));}}
 for (name,a,e) in [("position.x",c.position.x,vec(&t["origin"]).x),("position.y",c.position.y,vec(&t["origin"]).y),("velocity.x",c.velocity.x,vec(&s["velocity"]).x),("velocity.y",c.velocity.y,vec(&s["velocity"]).y),("angular",c.angular_velocity,s["angular"].as_f64().unwrap()),("basis.x",c.body_basis.0.x as f64,vec(&t["x"]).x),("basis.y",c.body_basis.0.y as f64,vec(&t["x"]).y)] {
  if (a as f32).to_bits()!=(e as f32).to_bits() {bad.insert(name.into(),json!([a,e]));}
 }
 for(name,a,e)in [("basis_y.x",c.body_basis.1.x as f64,vec(&t["y"]).x),("basis_y.y",c.body_basis.1.y as f64,vec(&t["y"]).y),("boost",c.boost_energy,s["boost"].as_f64().unwrap()),("acceleration.x",c.acceleration.x,vec(&s["acceleration"]).x),("acceleration.y",c.acceleration.y,vec(&s["acceleration"]).y)] {
  if (a as f32).to_bits()!=(e as f32).to_bits(){bad.insert(name.into(),json!([a,e]));}
 }
 for(i,(wheel,sw)) in c.wheels.iter().zip(s["wheels"].as_array().unwrap()).enumerate() {
  let a=wheel.angle_deg as f32;let e=sw["_currentRotationDegrees"].as_f64().unwrap() as f32;
  if a.to_bits()!=e.to_bits(){bad.insert(format!("wheel{i}"),json!([a as f64,e as f64]));}
  let e=vec(&sw["_lastPosition"]);let a=wheel.previous_position.unwrap_or_default();
  if (a.x as f32).to_bits()!=(e.x as f32).to_bits()||(a.y as f32).to_bits()!=(e.y as f32).to_bits(){bad.insert(format!("wheel{i}.history"),json!([[a.x,a.y],[e.x,e.y]]));}
 }
 if let Some(contacts)=s["contacts"].as_array(){
  if c.wall_contacts.len()!=contacts.len(){bad.insert("contact_count".into(),json!([c.wall_contacts.len(),contacts.len()]));}
  for(i,(a,e))in c.wall_contacts.iter().zip(contacts).enumerate(){
   for(key,a,e)in [("normal",a.normal,vec(&e["normal"])),("point",a.point,vec(&e["point"]))]{
    if (a.x as f32).to_bits()!=(e.x as f32).to_bits()||(a.y as f32).to_bits()!=(e.y as f32).to_bits(){bad.insert(format!("contact{i}.{key}"),json!([[a.x,a.y],[e.x,e.y]]));}
   }
  }
 }
 Value::Object(bad)
}
fn main() {
 rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
 let args:Vec<_>=std::env::args().collect();let scene:Value=serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
 let log=std::fs::read_to_string(&args[2]).unwrap();
 let states:Vec<Value>=log.lines().filter_map(|l|l.strip_prefix("FIDELITY_FRAME ")).map(|l|serde_json::from_str::<Value>(l).unwrap()).filter(|s|s["phase"]=="before").collect();
 let controls:Vec<Value>=log.lines().filter_map(|l|l.strip_prefix("FIDELITY_CONTROL ")).map(|l|serde_json::from_str::<Value>(l).unwrap()).collect();
 let mut w=World::from_scene(&scene);if args.len()<4 || args[3]!="wall" {w.track.shapes.clear();w.track.walls.clear();}
 let eliminate=args.iter().any(|a|a=="eliminate");
 let mut full=restore(&w,&states[0]); let mut rows=Vec::new();let mut exact=0;let mut full_exact=0;
 if scene["track"]["native_creation"]=="tilemap" {for _ in 0..8 {full.passive_step(&w,DT,eliminate);}}
 let reset_tick=scene["reset_tick"].as_u64().unwrap_or(90) as usize;
 for i in 0..states.len()-1 {
  let control=if controls.is_empty() {Controls{acceleration:if i<120 {1.0}else{0.0},steering:if i<30 {0.0}else if i<90 {0.4}else{-0.3},..Controls::default()}}
   else {let c=&controls[i]["output"];Controls{acceleration:c[0].as_f64().unwrap(),steering:c[1].as_f64().unwrap(),brake:c[2].as_f64().unwrap(),handbrake:c[3].as_f64().unwrap(),boost:c[4].as_f64().unwrap()}};
  let mut c=restore(&w,&states[i]);
  if i==90 && args.iter().any(|a|a=="deactivate") {c.deactivate();full.deactivate();}
  let reset=args.iter().any(|a|a=="reset");
  if i==reset_tick && reset {
   let position=scene.get("reset_position").map(|v|V2::from(altd_sim::curve::json_vector(v))).unwrap_or(V2::new(3840.0,1664.0));
   let rotation=scene["reset_rotation"].as_f64().unwrap_or(std::f32::consts::FRAC_PI_2 as f64);
   c.queue_reset(position,rotation);full.queue_reset(position,rotation);
  }
  if reset && (reset_tick..reset_tick+8).contains(&i) {c.passive_step(&w,DT,eliminate);full.passive_step(&w,DT,eliminate);}
  else {c.step(&w,&control,DT,eliminate);full.step(&w,&control,DT,eliminate);}
  let one=differences(&c,&states[i+1]);let continuous=differences(&full,&states[i+1]);
  if one.as_object().unwrap().is_empty(){exact+=1;}if continuous.as_object().unwrap().is_empty(){full_exact+=1;}
  rows.push(json!({"tick":i+1,"one":one,"continuous":continuous}));
 }
 println!("{}",json!({"steps":rows.len(),"one_step_exact":exact,"continuous_exact":full_exact,"rows":rows}));
}
