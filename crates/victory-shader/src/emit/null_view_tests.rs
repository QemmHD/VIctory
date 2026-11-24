//! What a host GPU does with a null descriptor whose kind or dimension is
//! not the one the shader declares (#392, #325, #548): the cases the host
//! heap makes today (`victory-gpu`'s `descriptor/host.rs`: every slot
//! starts as a null 2D texture SRV, a null buffer UAV is typed `R32_UINT`,
//! a null texture UAV is 2D), each next to the matching null descriptor.
//!
//! Plain HLSL compiled with the Windows SDK's `dxc.exe` (skipped without
//! it), run on the default adapter. Slot 0 holds the descriptor under
//! test, slot 1 a raw UAV the shader writes `0x100` plus what it read to.
//! D3D12 makes reads of a null descriptor zero and drops writes, so `0x100`
//! is the answer where the kinds match. Run each test in its own process
//! (`--exact`): a removed device takes the rest of a process's tests with
//! it.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::host::{Dispatch, Host, View};
use super::validator;

/// The Windows SDK's `dxc.exe` (next to the `dxil.dll` the validator uses),
/// or `VICTORY_PC_DXC`.
fn dxc() -> Option<PathBuf> {
    let p = match std::env::var_os("VICTORY_PC_DXC") {
        Some(p) => PathBuf::from(p),
        None => validator::Validator::default_path()?.with_file_name("dxc.exe"),
    };
    p.exists().then_some(p)
}

/// A `cs_6_6` program: `body` reads slot 0 into `v`; slot 1 gets `v`.
fn compile(dxc: &PathBuf, name: &str, body: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("victory-null-view-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{name}.hlsl"));
    let output = dir.join(format!("{name}.bin"));
    let text = format!(
        "[RootSignature(\"RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)\")]\n\
         [numthreads(1, 1, 1)]\n\
         void cs() {{\n\
             RWByteAddressBuffer o = ResourceDescriptorHeap[1];\n\
             uint v = 0x100;\n\
             {body}\n\
             o.Store(0, v);\n\
         }}\n"
    );
    std::fs::write(&input, text).unwrap();
    let out = std::process::Command::new(dxc)
        .args(["-T", "cs_6_6", "-E", "cs", "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(output).unwrap()
}

/// Runs `body` with `view` in slot 0: what slot 1 holds afterwards, or why
/// it could not run (a removed device names its reason).
fn case(name: &str, body: &str, view: View) -> Option<Result<u32, String>> {
    let Some(dxc) = dxc() else {
        eprintln!("no PC dxc.exe; skipped");
        return None;
    };
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return None;
    };
    let cs = compile(&dxc, name, body);
    // SAFETY: the blob is a DXIL container carrying its root signature.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &cs) }
        .unwrap_or_else(|e| panic!("root signature: {e}"));
    let pso = host
        .compute_pipeline_with(&root, &cs)
        .unwrap_or_else(|e| panic!("pipeline: {e}"));
    let got = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: Vec::new(),
                views: &[view, View::RawUav(4)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .map(|out| out[0][0]);
    // SAFETY: a plain device query.
    let removed = unsafe { host.device.GetDeviceRemovedReason() };
    eprintln!("{name}: {got:x?}, device {removed:?}");
    Some(got)
}

fn srv_texture(dimension: D3D12_SRV_DIMENSION) -> View<'static> {
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: dimension,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    if dimension == D3D12_SRV_DIMENSION_TEXTURE1D {
        desc.Anonymous.Texture1D = D3D12_TEX1D_SRV {
            MipLevels: 1,
            ..Default::default()
        };
    } else {
        desc.Anonymous.Texture2D = D3D12_TEX2D_SRV {
            MipLevels: 1,
            ..Default::default()
        };
    }
    View::NullSrv(desc)
}

/// `descriptor/host.rs`'s `null_view`: what every host heap slot starts as.
fn heap_default() -> View<'static> {
    srv_texture(D3D12_SRV_DIMENSION_TEXTURE2D)
}

fn srv_raw() -> View<'static> {
    View::NullSrv(D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    })
}

fn uav(format: DXGI_FORMAT, dimension: D3D12_UAV_DIMENSION, raw: bool) -> View<'static> {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ViewDimension: dimension,
        ..Default::default()
    };
    if dimension == D3D12_UAV_DIMENSION_BUFFER && raw {
        desc.Anonymous.Buffer = D3D12_BUFFER_UAV {
            NumElements: 1,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            ..Default::default()
        };
    } else if dimension == D3D12_UAV_DIMENSION_TEXTURE3D {
        desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
            WSize: u32::MAX,
            ..Default::default()
        };
    }
    View::NullUav(desc)
}

const RW_RAW: &str = "RWByteAddressBuffer b = ResourceDescriptorHeap[0]; \
                      b.Store(0, 5u); v += b.Load(4);";
const RW_3D: &str = "RWTexture3D<uint> t = ResourceDescriptorHeap[0]; \
                     t[uint3(0, 0, 0)] = 5u; v += t[uint3(1, 2, 3)];";
const RW_2D: &str = "RWTexture2D<uint> t = ResourceDescriptorHeap[0]; \
                     t[uint2(0, 0)] = 5u; v += t[uint2(1, 2)];";
const RAW: &str = "ByteAddressBuffer b = ResourceDescriptorHeap[0]; v += b.Load(4);";
const TEX_2D: &str = "Texture2D<float4> t = ResourceDescriptorHeap[0]; \
                      v += (uint)(t.Load(int3(1, 1, 0)).x * 1000.0 + 0.5);";

fn expect_zero_read(r: Option<Result<u32, String>>) {
    if let Some(r) = r {
        assert_eq!(r, Ok(0x100));
    }
}

// Controls: the null descriptor's kind matches the declaration.

#[test]
fn control_raw_uav_through_a_null_raw_uav() {
    let view = uav(DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER, true);
    expect_zero_read(case("control_raw_uav", RW_RAW, view));
}

#[test]
fn control_3d_uav_through_a_null_3d_uav() {
    let view = uav(DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_TEXTURE3D, false);
    expect_zero_read(case("control_3d_uav", RW_3D, view));
}

#[test]
fn control_raw_srv_through_a_null_raw_srv() {
    expect_zero_read(case("control_raw_srv", RAW, srv_raw()));
}

#[test]
fn control_2d_srv_through_a_null_2d_srv() {
    expect_zero_read(case("control_2d_srv", TEX_2D, heap_default()));
}

// Mismatches the host heap makes today.

/// #392's third hang: a UAV slot the title never wrote is still the heap's
/// initial null 2D texture SRV.
#[test]
fn raw_uav_through_the_heaps_initial_null_srv() {
    expect_zero_read(case("raw_uav_via_null_srv2d", RW_RAW, heap_default()));
}

/// A 2D texture UAV slot never written: image store through a null SRV.
#[test]
fn texture_uav_through_the_heaps_initial_null_srv() {
    expect_zero_read(case("uav2d_via_null_srv2d", RW_2D, heap_default()));
}

/// `null_for` gives buffer UAVs a typed `R32_UINT` null view; translated
/// shaders access buffers raw.
#[test]
fn raw_uav_through_a_typed_null_uav() {
    let view = uav(DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_BUFFER, false);
    expect_zero_read(case("raw_uav_via_typed_null_uav", RW_RAW, view));
}

/// `null_for` before #205: a 2D null UAV where the shader writes 3D.
#[test]
fn texture_3d_uav_through_a_null_2d_uav() {
    let view = uav(DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_TEXTURE2D, false);
    expect_zero_read(case("uav3d_via_null_uav2d", RW_3D, view));
}

/// A buffer SRV slot of unknown kind keeps the null 2D texture SRV.
#[test]
fn raw_srv_through_the_heaps_initial_null_srv() {
    expect_zero_read(case("raw_srv_via_null_srv2d", RAW, heap_default()));
}

/// GPU-based validation's id 940 (#548, #325): a 2D read through a 1D view.
#[test]
fn texture_2d_srv_through_a_null_1d_srv() {
    let view = srv_texture(D3D12_SRV_DIMENSION_TEXTURE1D);
    expect_zero_read(case("srv2d_via_null_srv1d", TEX_2D, view));
}
