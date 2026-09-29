use altd_sim::random_track::{self,Connection,TrackGenerator,RandomTrackConfig};
use serde_json::{Value,json};
fn main(){
 let arg=std::env::args().nth(1).unwrap();
 if arg=="search" {
  let mut sets:Vec<Vec<Connection>>=Vec::new();
  for b in random_track::blocks(){
   let mut cs=b.connections.clone();for c in &mut cs{c.kind=0;}
   cs.sort_by_key(|c|(c.side,c.index));
   if !sets.contains(&cs){sets.push(cs);}
  }
  for a in 0..4 {for b in a+1..4 {for i in 0..3 {for j in 0..3 {
   let cs=vec![Connection{side:a,index:i,kind:0},Connection{side:b,index:j,kind:0}];
   if !sets.contains(&cs){sets.push(cs);}
  }}}}
  let end:u32=std::env::args().nth(2).unwrap_or("200000".into()).parse().unwrap();
  for seed in 0..end {
   let mut hashes:Vec<_>=sets.iter().enumerate().map(|(i,c)|(random_track::connection_set_hash(seed,c),i)).collect();
   hashes.sort_unstable();
   if let Some(pair)=hashes.windows(2).find(|p|p[0].0==p[1].0){println!("{}",json!({"seed":seed,"hash":pair[0].0,"sets":[sets[pair[0].1],sets[pair[1].1]]}));return;}
  }
  println!("{}",json!({"seeds_checked":end,"sets":sets.len(),"collision":null}));return;
 }
 let data:Value=serde_json::from_str(&std::fs::read_to_string(arg).unwrap()).unwrap();
 verify(&data);println!("{}",json!({"seed":data["catalog"]["hashcode_seed"],"cases":data["cases"].as_array().unwrap().len(),"exact":true}));
}
pub fn verify(data:&Value) {
 let catalog=&data["catalog"];let seed=catalog["hashcode_seed"].as_u64().unwrap()as u32;
 for row in catalog["connection_hashes"].as_array().unwrap(){assert_eq!(random_track::combine_hash(seed,row["side"].as_i64().unwrap()as i32,row["index"].as_i64().unwrap()as i32),row["hash"].as_i64().unwrap()as i32);}
 for (block,want) in random_track::blocks().iter().zip(catalog["block_hashes"].as_array().unwrap()) {
  assert_eq!(random_track::connection_set_hash(seed,&block.connections),want.as_i64().unwrap()as i32);
 }
 let generator=TrackGenerator::new(seed);
 assert_eq!(generator.lookup_json(),catalog["lookups"]);
 for case in data["cases"].as_array().unwrap(){
  let mut config:RandomTrackConfig=serde_json::from_value(case["config"].clone()).unwrap();
  let mut state=serde_json::from_value(case["state"].clone()).unwrap();
  let result=generator.generate(&mut config,&mut state);
  if let Some(error)=case["error"].as_str(){assert_eq!(result.unwrap_err(),error);}else{assert_eq!(serde_json::to_value(result.unwrap()).unwrap(),case["result"]);}
  assert_eq!(serde_json::to_value(state).unwrap(),case["after"]);
  assert_eq!(serde_json::to_value(config).unwrap(),case["input_after"]);
 }
}
