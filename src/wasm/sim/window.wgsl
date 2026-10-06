// Full GPU simulation, with separate kernels to keep shader compilation bounded.
// Each (car, sensor) pair senses in its own invocation, sensor-major in blocks
// of 32 cars so a workgroup evaluates one sensor kind without divergence. The
// value goes to the car's input slots and its domain-error bits to a
// per-sensor word after the input area, which the forward kernel folds into
// the car's error field.
fn sensor_errors(index: u32) -> u32 { return 4u+params.population*128u+index*64u; }
fn sense(index: u32, sensor: u32) {
    var err = 0u;
    var here = 0.0f;
    let desc = load_SensorDesc(params.sensors_offset+sensor*6u);
    if (desc.kind == S_CORRECT_DIRECTION || desc.kind == S_TRACK_CURVATURE) { here = sensor_path_offset(); }
    let value = sensor_value(desc, here, &err);
    let at=4u+index*128u+sensor*2u;
    results[at]=value.x;results[at+1u]=value.y;
    results[sensor_errors(index)+sensor]=err;
}
fn statistics(k: u32) {
    var err=0u;
    let first=select(params.stats_phase,6u,params.stats_phase==0u);
    update_stats(U64(params.start_tick+k,0u),U64(params.start_tick+k-first,0u),params.eliminate_idle != 0u,&err);
    sim_car.error=sim_car.error | err;
}
fn time_limit(k: u32) -> bool {
    if (params.has_limit==0u) { return false; }
    let first=select(params.stats_phase,6u,params.stats_phase==0u);
    let updates=(params.start_tick+k-first)/6u;
    return f64_ge(f64_mul(f64_from_u32(updates),F64(0x9999999au,0x3fb99999u)),params.limit);
}
fn load_car_state(index: u32) {
    sim_world=load_World(params.world_offset);
    sim_vehicle=load_VehicleDesc(params.vehicle_offset);
    load_Car(index*800u);
    load_Agent(index*94u);
}
fn save_car_state(index: u32) {
    store_Car(index*800u);
    store_Agent(index*94u);
}

fn eligible(index:u32) -> bool {
 let batch_size=(params.population+7u)/8u;
 let batch=(params.start_batch+((params.reserved_a-1u)%8u)*params.batches_per_tick)%8u;
 return (sim_car.flags & CAR_ACTIVE) != 0u && (index/batch_size+8u-batch)%8u<params.batches_per_tick;
}
@compute @workgroup_size(32)
fn sensors_kernel(@builtin(global_invocation_id) id:vec3<u32>) {
 let stride=(params.population+31u)/32u*32u;
 let i=id.x%stride;let s=id.x/stride;
 if(i>=params.population || s>=params.sensor_count || results[1]!=0u){return;}
 sim_world=load_World(params.world_offset);
 sim_vehicle=load_VehicleDesc(params.vehicle_offset);
 load_sensor_Car(i*800u);
 if(eligible(i)){sense(i,s);}
}
var<workgroup> layer_input: array<F64,64>;
var<workgroup> sensor_error: atomic<u32>;
@compute @workgroup_size(64)
fn forward_kernel(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_id) lane:vec3<u32>) {
 let i=group.x;let j=lane.x;
 var enabled=false;
 if(j==0u){atomicStore(&sensor_error,0u);}
 if(i<params.population && results[1]==0u){
  sim_car.flags=cars[i*800u+16u];enabled=eligible(i);
 }
 workgroupBarrier();
 if(enabled && j<params.sensor_count){
  let at=4u+i*128u+j*2u;layer_input[j]=F64(results[at],results[at+1u]);
  let err=results[sensor_errors(i)+j];
  if(err!=0u){atomicOr(&sensor_error,err);}
 }
 workgroupBarrier();
 if(enabled && j==0u){cars[i*800u+39u]=cars[i*800u+39u] | atomicLoad(&sensor_error);}
 var position=i*params.stride;
 let math=load_World(params.world_offset).math_profile;
 for(var layer=0u;layer+1u<params.layers;layer=layer+1u){
  let rows=world_data[params.shape_offset+layer];
  let cols=world_data[params.shape_offset+layer+1u];
  var output=F64_ZERO;
  if(enabled && j<cols){
   var sum=F64_ZERO;
   for(var r=0u;r<rows;r=r+1u){sum=f64_add(sum,f64_mul(layer_input[r],networks[position+r*cols+j]));}
   output=profile_tanh(math,f64_add(sum,networks[position+rows*cols+j]));
  }
  workgroupBarrier();
  layer_input[j]=output;
  workgroupBarrier();
  position=position+(rows+1u)*cols;
 }
 if(enabled && j<5u){
  let src=i32(world_data[params.control_offset+j]);
  var control=F64_ZERO;if(src>=0){control=layer_input[u32(src)];}
  agents[i*94u+j*2u]=control.x;agents[i*94u+j*2u+1u]=control.y;
 }
}
@compute @workgroup_size(32)
fn stats_kernel(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=params.population || results[1]!=0u){return;}
 load_car_state(i);statistics(params.reserved_a);save_car_state(i);
}
@compute @workgroup_size(1)
fn stop_kernel() {
 if(results[1]!=0u){return;}
 var live=false;
 for(var i=0u;i<params.population;i=i+1u){live=live || (cars[i*800u+16u]&CAR_ACTIVE)!=0u;}
 if(time_limit(params.reserved_a) || (params.stop_inactive!=0u && !live)){
   results[0]=params.reserved_a;results[1]=1u;
 }
}
@compute @workgroup_size(32)
fn reset_kernel(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=params.population){return;}
 agents[i*94u+92u]=agents[i*94u+92u] & ~AGENT_HAS_DEACTIVATED;
 agents[i*94u+90u]=0u;agents[i*94u+91u]=0u;
}
@compute @workgroup_size(32)
fn drive_kernel(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=params.population || results[1]!=0u){return;}
 load_car_state(i);
 let carry=step_begin(sim_agent.controls,true);
 store_StepCarry(4u+i*128u,carry);store_Car(i*800u);
}
@compute @workgroup_size(32)
fn step_kernel(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=params.population || results[1]!=0u){return;}
 load_car_state(i);
 let was_active=(sim_car.flags & CAR_ACTIVE)!=0u;
 let contact=step_end(load_StepCarry(4u+i*128u),true,params.eliminate_wall!=0u);
 if(contact){sim_agent.flags=sim_agent.flags | AGENT_PENDING_CONTACT;}
 if(was_active && (sim_car.flags & CAR_ACTIVE)==0u){
  sim_agent.flags=sim_agent.flags | AGENT_HAS_DEACTIVATED;
  sim_agent.deactivated_at=U64(params.start_tick+params.reserved_a,0u);
 }
 save_car_state(i);
}
