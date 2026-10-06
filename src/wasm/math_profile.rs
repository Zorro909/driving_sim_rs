//! The math profile of the browser's machine (crate::math::profile).
//!
//! The Windows build is a high-entropy User-Agent Client Hint, which only an
//! asynchronous call returns. Module initialization starts one cached query;
//! the web package awaits it before its asynchronous initializer resolves.
//! Synchronous hosts guess from the platform until the query has finished.

use crate::math::profile::MathProfile;
use js_sys::{Array, Function, Promise, Reflect};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

/// What `detectMathProfile` found: 0 until it finished, else the profile's
/// index plus one.
static DETECTED: AtomicU32 = AtomicU32::new(0);

thread_local! {
    /// The promise is local to a JavaScript realm; the result above is shared
    /// with Rayon workers in a threaded package.
    static DETECTION: RefCell<Option<Promise>> = const { RefCell::new(None) };
}

fn get(object: &JsValue, key: &str) -> Option<JsValue> {
    Reflect::get(object, &key.into())
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null())
}

/// `navigator.userAgentData`, present in Chromium browsers and their workers.
fn user_agent_data() -> Option<JsValue> {
    get(&get(&js_sys::global(), "navigator")?, "userAgentData")
}

fn on_windows() -> bool {
    if let Some(platform) = user_agent_data().and_then(|data| get(&data, "platform")?.as_string()) {
        return platform == "Windows";
    }
    let navigator = get(&js_sys::global(), "navigator");
    navigator
        .and_then(|n| get(&n, "userAgent")?.as_string())
        .is_some_and(|agent| agent.contains("Windows"))
}

/// The profile of a Windows `platformVersion` hint, whose major component is
/// the Windows.Foundation.UniversalApiContract version: 19 arrived with
/// Windows 11 24H2 (build 26100), 23H2 still reports 15.
fn for_platform_version(version: &str) -> Option<MathProfile> {
    let major: u32 = version.split('.').next()?.parse().ok()?;
    Some(if major >= 19 {
        MathProfile::Win11Fma3
    } else {
        MathProfile::Win10Fma3
    })
}

/// The profile of sessions whose options name none. Windows without a known
/// version is taken for a current Windows 11.
pub(crate) fn browser_math_profile() -> MathProfile {
    match DETECTED.load(Ordering::Relaxed) {
        0 if on_windows() => MathProfile::Win11Fma3,
        0 => MathProfile::Proton,
        n => MathProfile::ALL[n as usize - 1],
    }
}

async fn platform_version() -> Option<String> {
    let data = user_agent_data()?;
    let method: Function = get(&data, "getHighEntropyValues")?.dyn_into().ok()?;
    let hints = Array::of1(&"platformVersion".into());
    let promise: Promise = method.call1(&data, &hints).ok()?.dyn_into().ok()?;
    get(&JsFuture::from(promise).await.ok()?, "platformVersion")?.as_string()
}

/// Detects the math profile of this machine, which later sessions without a
/// `mathProfile` option use, and resolves to its name. Initialization and
/// later calls share one promise. Browsers without User-Agent Client Hints
/// (Firefox, Safari) cannot tell Windows 10 from 11.
#[wasm_bindgen(js_name = detectMathProfile)]
pub fn detect_math_profile() -> Promise {
    DETECTION.with(|cached| {
        cached
            .borrow_mut()
            .get_or_insert_with(|| {
                future_to_promise(async {
                    let profile = match DETECTED.load(Ordering::Relaxed) {
                        0 if on_windows() => match platform_version().await {
                            Some(version) => for_platform_version(&version).unwrap_or(MathProfile::Win11Fma3),
                            None => MathProfile::Win11Fma3,
                        },
                        0 => MathProfile::Proton,
                        n => MathProfile::ALL[n as usize - 1],
                    };
                    DETECTED.store(profile.index() + 1, Ordering::Relaxed);
                    Ok(profile.name().into())
                })
            })
            .clone()
    })
}

/// The names of every math profile.
#[wasm_bindgen(js_name = mathProfiles)]
pub fn math_profiles() -> Vec<String> {
    MathProfile::ALL.iter().map(|p| p.name().to_owned()).collect()
}

/// A `mathProfile` argument: a profile name, or undefined for the default.
pub(crate) fn math_profile_arg(name: Option<String>) -> Result<MathProfile, JsError> {
    match name {
        Some(name) => name.parse().map_err(|e: String| JsError::new(&e)),
        None => Ok(browser_math_profile()),
    }
}
