//! Godot 4.3 Curve2D baking and linear sampling, preserving float operation order.
//! Adapted from Godot Engine scene/resources/curve.cpp and core/math/vector2.h.
//! Copyright (c) 2014-present Godot Engine contributors; MIT license.
use crate::godot_math::F2;
use crate::native_math;
use serde_json::Value;
#[derive(Clone,Debug)]
pub struct Curve {pub points:Vec<F2>,pub offsets:Vec<f32>,pub forwards:Vec<F2>,grid:Option<crate::segment_grid::SegmentGrid>}
fn divide(p:F2,n:f32)->F2 {F2{x:p.x/n,y:p.y/n}}
fn bezier([a,b,c,d]:[F2;4],t:f32)->F2 {
 let omt=1.0-t;let omt2=omt*omt;let omt3=omt2*omt;let t2=t*t;let t3=t2*t;
 a*omt3+b*omt2*t*3.0+c*omt*t2*3.0+d*t3
}
fn approximate(a:f32,b:f32)->bool {a==b || (a-b).abs()<(1e-5*a.abs()).max(1e-5)}
fn tangent([a,b,c,d]:[F2;4],t:f32)->F2 {
 if (t.abs()<1e-5 && approximate(a.x,b.x)&&approximate(a.y,b.y)) || ((t-1.0).abs()<1e-5 && approximate(c.x,d.x)&&approximate(c.y,d.y)) {return (d-a).normalized();}
 let omt=1.0-t;let omt2=omt*omt;let t2=t*t;
 ((b-a)*3.0*omt2+(c-b)*6.0*omt*t+(d-c)*3.0*t2).normalized()
}
fn subdivide(p:[F2;4],begin:f32,end:f32,depth:u32,interval:f32,out:&mut Vec<(f32,F2)>) {
 if (bezier(p,begin)-bezier(p,end)).length()>interval && depth<10 {
  let mid=(begin+end)*0.5;
  subdivide(p,begin,mid,depth+1,interval,out);
  out.push((mid,bezier(p,mid)));
  subdivide(p,mid,end,depth+1,interval,out);
 }
}
pub fn json_vector(v:&Value)->F2 {
 if v.is_array() {F2{x:v[0].as_f64().unwrap() as f32,y:v[1].as_f64().unwrap() as f32}}
 else {F2{x:v["x"].as_f64().unwrap() as f32,y:v["y"].as_f64().unwrap() as f32}}
}
impl Curve {
 pub fn from_json(data:&Value)->Self {
  let controls=data["points"].as_array().unwrap();let interval=data["bake_interval"].as_f64().unwrap() as f32;
  let mut curve=Self{points:Vec::new(),offsets:Vec::new(),forwards:Vec::new(),grid:None};
  for (i,pair) in controls.windows(2).enumerate() {
   let a=json_vector(&pair[0]["position"]);let d=json_vector(&pair[1]["position"]);
   let p=[a,a+json_vector(&pair[0]["outgoing"]),d+json_vector(&pair[1]["incoming"]),d];
   if i==0 {curve.points.push(a);curve.forwards.push(tangent(p,0.0));}
   let mut mid=Vec::new();subdivide(p,0.0,1.0,0,interval,&mut mid);
   for (t,pos) in mid {curve.points.push(pos);curve.forwards.push(tangent(p,t));}
   curve.points.push(d);curve.forwards.push(tangent(p,1.0));
  }
  let mut distance=0.0;curve.offsets.push(distance);
  for p in curve.points.windows(2) {distance+=(p[0]-p[1]).length();curve.offsets.push(distance);}
  curve.grid=crate::segment_grid::SegmentGrid::new(&curve.points);
  curve
 }
 pub fn length(&self)->f32 {*self.offsets.last().unwrap_or(&0.0)}
 fn interval(&self,offset:f32)->(usize,f32) {
  let offset=offset.clamp(0.0,self.length());let(mut start,mut end)=(0,self.points.len());let mut index=(start+end)/2;
  while start<index {if offset<=self.offsets[index] {end=index;}else{start=index;}index=(start+end)/2;}
  let span=self.offsets[index+1]-self.offsets[index];
  (index,if span<f32::EPSILON {0.5}else{(offset-self.offsets[index])/span})
 }
 pub fn direction(&self,offset:f32)->F2 {
  let(i,t)=self.interval(offset);let a=self.forwards[i];let b=self.forwards[i+1];
  let a2=a.dot(a);let b2=b.dot(b);
  if a2==0.0 || b2==0.0 {return (a+(b-a)*t).normalized();}
  let length=a2.sqrt();let result_length=length+(b2.sqrt()-length)*t;
  let angle=native_math::atan2(a.cross(b),a.dot(b))*t;
  let(s,c)=(native_math::engine_sin(angle),native_math::engine_cos(angle));
  (F2{x:a.x*c-a.y*s,y:a.x*s+a.y*c}*(result_length/length)).normalized()
 }
 pub fn position(&self,offset:f32)->F2 {let(i,t)=self.interval(offset);self.points[i]+(self.points[i+1]-self.points[i])*t}
 fn project(&self,query:F2,i:usize)->(f32,f32) {
  let span=self.offsets[i+1]-self.offsets[i];let origin=self.points[i];let direction=divide(self.points[i+1]-origin,span);
  let d=(query-origin).dot(direction).clamp(0.0,span);let projection=origin+direction*d;
  let delta=projection-query;(self.offsets[i]+d,delta.dot(delta))
 }
 pub fn closest_offset(&self,query:F2)->f32 {
  let(mut nearest,mut distance)=(0.0,-1.0);
  let mut visit=|i:usize| {let(offset,dist)=self.project(query,i);if distance<0.0 || dist<distance {nearest=offset;distance=dist;}};
  // Segment 0 is always accepted first (a NaN there sticks), then the exact
  // candidate subset in ascending order; see segment_grid.
  let count=self.points.len()-1;
  if count>0 {
   if let Some(grid)=&self.grid {
    visit(0);
    // Revisiting segment 0 in the fallback cannot change the result.
    if grid.scan(query,|i|self.project(query,i).1,|i|if i>0 {visit(i)}) {return nearest;}
   }
  }
  for i in 0..count {visit(i);}
  nearest
 }
}
