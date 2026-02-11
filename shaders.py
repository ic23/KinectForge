# =============================
# GLSL Шейдеры
# =============================

VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;
out float v_depth_t;  // normalized depth for cyber gradient
out float v_row;      // for scanlines

uniform mat4 mvp;
uniform float point_size;
uniform float time;

// Cyberspace uniforms
uniform int cyber_enabled;
uniform int cyber_jitter;
uniform int cyber_glitch_bands;
uniform float depth_min;
uniform float depth_max;
uniform float cluster_threshold;
uniform int depth_scale_points;
uniform float depth_scale_factor;

// Render mode: 0=normal, 1=noise particles, 2=grid
uniform int render_mode;

// New effects
uniform int wave_distortion;
uniform float wave_amplitude;
uniform float wave_frequency;
uniform int voxelize;
uniform float voxel_size;
uniform int pulse;
uniform float pulse_speed;

// PCG hash — better quality, no visible patterns or temporal correlation
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float hash(vec3 p) {
    uvec3 u = floatBitsToUint(p);
    uint h = pcg(u.x ^ pcg(u.y ^ pcg(u.z)));
    return float(h) / 4294967295.0;
}

void main() {
    vec3 pos = in_position;
    
    // Noise particle animation (render_mode == 1)
    if (render_mode == 1) {
        float t = time * 0.3;
        float h1 = hash(in_position * 17.31);
        float h2 = hash(in_position.yzx * 23.17);
        float h3 = hash(in_position.zxy * 31.73);
        pos.x += sin(h1 * 6.283 + t) * 0.08;
        pos.y += cos(h2 * 6.283 + t * 0.7) * 0.05;
        pos.z += sin(h3 * 6.283 + t * 0.5) * 0.05;
    }
    
    // Jitter effect
    if (cyber_jitter == 1 && render_mode == 0) {
        float n = hash(pos + vec3(time * 0.1));
        pos += (vec3(n, hash(pos.yzx + time), hash(pos.zxy + time)) - 0.5) * 0.003;
    }
    
    // Glitch bands - horizontal displacement
    if (cyber_glitch_bands == 1 && render_mode == 0) {
        float glitch_trigger = step(0.96, hash(vec3(floor(time * 25.0), 0.0, 0.0)));
        if (glitch_trigger > 0.5) {
            float band_y = hash(vec3(floor(time * 7.0), 1.0, 0.0)) * 2.0 - 1.0;
            float band_h = 0.05 + hash(vec3(floor(time * 11.0), 2.0, 0.0)) * 0.1;
            float in_band = step(abs(pos.y - band_y), band_h);
            float shift = (hash(vec3(floor(time * 13.0), 3.0, 0.0)) - 0.5) * 0.16;
            pos.x += shift * in_band;
        }
    }
    
    // Wave distortion
    if (wave_distortion == 1 && render_mode == 0) {
        float wave = sin(pos.y * wave_frequency * 20.0 + time * 3.0) * wave_amplitude;
        float wave2 = cos(pos.z * wave_frequency * 15.0 + time * 2.0) * wave_amplitude * 0.5;
        pos.x += wave;
        pos.y += wave2;
    }
    
    // Voxelize — snap to grid
    if (voxelize == 1 && render_mode == 0) {
        pos = floor(pos / voxel_size + 0.5) * voxel_size;
    }
    
    gl_Position = mvp * vec4(pos, 1.0);
    
    // Compute depth gradient (0 = near, 1 = far)
    float z_val = -in_position.z;
    v_depth_t = clamp((z_val - depth_min) / max(depth_max - depth_min, 0.01), 0.0, 1.0);
    
    // Point size with optional perspective depth scaling
    float final_size = point_size;
    if (depth_scale_points == 1) {
        float scale = 1.0 + (1.0 - v_depth_t) * depth_scale_factor;
        final_size *= scale;
    }
    
    // Pulse — rhythmic size oscillation
    if (pulse == 1) {
        float p = 0.7 + 0.3 * sin(time * pulse_speed * 3.14159);
        final_size *= p;
    }
    gl_PointSize = final_size;
    
    v_row = gl_Position.y;
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
in float v_depth_t;
in float v_row;

out vec4 fragColor;

uniform float time;
uniform int cyber_enabled;
uniform int cyber_invert;
uniform int cyber_glitch_color;
uniform float cluster_threshold;  // normalized cluster split (0..1)
uniform int point_shape;           // 0 = circle, 1 = square
uniform int ssao_enabled;
uniform float ssao_strength;
uniform int render_mode;           // 0=normal, 1=noise, 2=grid
uniform int color_palette;          // 0=off, 1=thermal, 2=nightvision, 3=retro

// PCG hash — better quality, no visible patterns
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float hash(float n) {
    uint h = pcg(floatBitsToUint(n));
    return float(h) / 4294967295.0;
}

void main() {
    // Point shape
    if (point_shape == 0) {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25)
            discard;
    }
    
    vec3 color = v_color;
    
    // Color palettes (applied before cyberspace, can stack)
    if (color_palette > 0 && render_mode == 0) {
        float lum = dot(v_color, vec3(0.299, 0.587, 0.114));
        if (color_palette == 1) {
            // Thermal: blue → cyan → green → yellow → red → white
            float t = clamp(lum, 0.0, 1.0);
            if (t < 0.2) {
                color = mix(vec3(0.0, 0.0, 0.3), vec3(0.0, 0.4, 0.8), t / 0.2);
            } else if (t < 0.4) {
                color = mix(vec3(0.0, 0.4, 0.8), vec3(0.0, 0.8, 0.2), (t - 0.2) / 0.2);
            } else if (t < 0.6) {
                color = mix(vec3(0.0, 0.8, 0.2), vec3(0.9, 0.9, 0.0), (t - 0.4) / 0.2);
            } else if (t < 0.8) {
                color = mix(vec3(0.9, 0.9, 0.0), vec3(1.0, 0.2, 0.0), (t - 0.6) / 0.2);
            } else {
                color = mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.8) / 0.2);
            }
        } else if (color_palette == 2) {
            // Night vision: green phosphor
            float bright = lum * 1.4;
            float noise = hash(gl_FragCoord.x * 0.1 + gl_FragCoord.y * 0.01 + time * 5.0) * 0.08;
            color = vec3(bright * 0.15, bright * 0.95 + noise, bright * 0.12);
        } else if (color_palette == 3) {
            // Retro amber monochrome
            color = vec3(lum * 1.0, lum * 0.7, lum * 0.2);
        }
    }
    
    if (cyber_enabled == 1 && render_mode != 2) {
        // Depth value
        float raw_t = cyber_invert == 1 ? v_depth_t : 1.0 - v_depth_t;
        
        // Cluster-based split: 0 = near cluster, 1 = far cluster
        float ct = cyber_invert == 1 ? cluster_threshold : 1.0 - cluster_threshold;
        float is_far = step(ct, raw_t);
        
        // Smooth blend near the boundary (thin transition zone)
        float edge_soft = 0.04;
        float blend_t = smoothstep(ct - edge_soft, ct + edge_soft, raw_t);
        
        // Within-cluster local gradient for subtle variation
        float local_t;
        if (raw_t < ct) {
            local_t = raw_t / max(ct, 0.001);                     // 0..1 inside near cluster
        } else {
            local_t = (raw_t - ct) / max(1.0 - ct, 0.001);       // 0..1 inside far cluster
        }
        
        // Calculate luminance from original color
        float lum = dot(v_color, vec3(0.299, 0.587, 0.114));
        float intensity = 0.45 + 0.55 * lum;
        
        // Near cluster: Cyan tones with subtle local gradient
        vec3 near_color;
        near_color.r = (0.02 + local_t * 0.10) * intensity;
        near_color.g = (0.75 + local_t * 0.15) * intensity;
        near_color.b = (0.90 - local_t * 0.10) * intensity;
        
        // Far cluster: Magenta/hot pink tones with local gradient
        vec3 far_color;
        far_color.r = (0.70 + local_t * 0.25) * intensity;
        far_color.g = (0.08 + local_t * 0.12) * intensity;
        far_color.b = (0.55 - local_t * 0.15) * intensity;
        
        // Mix between clusters with smooth edge
        vec3 cyber_color = mix(near_color, far_color, blend_t);
        
        // Scanlines
        float scanline = mod(gl_FragCoord.y, 2.0) < 1.0 ? 1.0 : 0.75;
        cyber_color *= scanline;
        
        // Color glitch
        if (cyber_glitch_color == 1) {
            float glitch = step(0.96, hash(floor(time * 60.0)));
            if (glitch > 0.5) {
                float glitch_type = floor(hash(floor(time * 30.0) + 1.0) * 4.0);
                float frag_id = hash(gl_FragCoord.x * 0.01 + gl_FragCoord.y * 0.001);
                float in_glitch = step(0.7, hash(floor(time * 17.0) + frag_id));
                if (in_glitch > 0.5) {
                    if (glitch_type < 1.0) {
                        cyber_color = vec3(0.0, 0.9, 1.0);
                    } else if (glitch_type < 2.0) {
                        cyber_color = vec3(1.0, 0.0, 0.6);
                    } else if (glitch_type < 3.0) {
                        cyber_color = vec3(1.0, 1.0, 1.0);
                    } else {
                        cyber_color = cyber_color.bgr;
                    }
                }
            }
        }
        
        color = cyber_color;
    }
    
    // Point shading (hemisphere AO)
    if (ssao_enabled == 1) {
        vec2 pc = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(pc, pc);
        float sphere_shade = sqrt(max(1.0 - r2, 0.0));
        float ao = mix(1.0 - ssao_strength * 0.5, 1.0, sphere_shade);
        color *= ao;
    }
    
    // Pulse — brightness oscillation
    if (color_palette == 0) {  // only when no palette override
        // (pulse brightness is handled via point size in vertex shader)
    }
    
    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# =============================
# Fullscreen quad vertex shader (shared by post-processing passes)
# =============================

FULLSCREEN_VERTEX_SHADER = """
#version 330

in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

# =============================
# Temporal accumulation shader (ghost render + render trails)
# =============================

ACCUM_FRAGMENT_SHADER = """
#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D current_scene;
uniform sampler2D prev_accum;
uniform int ghost_enabled;
uniform int drip_enabled;
uniform float ghost_decay;   // how fast ghosts fade (e.g. 0.92)
uniform float drip_speed;    // vertical UV shift per frame
uniform vec3 bg_color;       // background color to detect empty pixels

void main() {
    vec3 current = texture(current_scene, v_uv).rgb;
    
    // Is this pixel actual content (not background)?
    float diff_from_bg = length(current - bg_color);
    bool is_content = diff_from_bg > 0.05;
    
    if (ghost_enabled == 0 && drip_enabled == 0) {
        fragColor = vec4(current, 1.0);
        return;
    }
    
    vec2 prev_uv = v_uv;
    if (drip_enabled == 1) {
        prev_uv.y += drip_speed;
    }
    
    // Sample previous accumulation; default to bg if out of bounds
    vec3 prev = bg_color;
    if (prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
        prev = texture(prev_accum, prev_uv).rgb;
    }
    
    // Decay only the trail portion (delta from background)
    vec3 trail = prev - bg_color;
    trail *= ghost_decay;
    
    // Hard cutoff: kill trail when too faint to prevent lingering artifacts
    if (length(trail) < 0.04) {
        trail = vec3(0.0);
    }
    
    vec3 faded = bg_color + trail;
    
    if (is_content) {
        fragColor = vec4(current, 1.0);
    } else if (length(trail) > 0.0) {
        fragColor = vec4(faded, 1.0);
    } else {
        fragColor = vec4(bg_color, 1.0);
    }
}
"""

# =============================
# Final composite shader (bloom + double glitch)
# =============================

COMPOSITE_FRAGMENT_SHADER = """
#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D scene_tex;
uniform int bloom_enabled;
uniform float bloom_strength;
uniform vec2 texel_size;
uniform int double_glitch_enabled;
uniform float time;
uniform int chromatic_aberration;
uniform float chromatic_strength;
uniform int edge_glow;
uniform float edge_glow_strength;
uniform int pixelate_enabled;
uniform float pixelate_size;

// PCG hash
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float hash(float n) {
    uint h = pcg(floatBitsToUint(n));
    return float(h) / 4294967295.0;
}

void main() {
    vec2 uv = v_uv;
    
    // Pixelate
    if (pixelate_enabled == 1) {
        vec2 grid = vec2(pixelate_size) * texel_size;
        uv = floor(uv / grid + 0.5) * grid;
    }
    
    vec3 color;
    
    // Chromatic aberration — sample R/G/B at offset UVs
    if (chromatic_aberration == 1) {
        float offset = chromatic_strength * 0.003;
        // Direction: radial from center
        vec2 dir = uv - 0.5;
        float r = texture(scene_tex, uv + dir * offset).r;
        float g = texture(scene_tex, uv).g;
        float b = texture(scene_tex, uv - dir * offset).b;
        color = vec3(r, g, b);
    } else {
        color = texture(scene_tex, uv).rgb;
    }
    
    // Double glitch — screen-space duplication
    if (double_glitch_enabled == 1) {
        float trigger = step(0.975, hash(floor(time * 25.0)));
        if (trigger > 0.5) {
            vec2 offset = vec2(
                (hash(floor(time * 37.0)) - 0.5) * 0.02,
                (hash(floor(time * 41.0)) - 0.5) * 0.02
            );
            vec3 dupe = texture(scene_tex, uv + offset).rgb;
            float dupe_bright = hash(floor(time * 53.0)) * 0.5 + 0.5;
            color = max(color, dupe * dupe_bright);
        }
    }
    
    // Bloom — optimized: 2 rings × 8 directions = 16 fetches (was 40)
    // Uses gaussian-like weights and higher brightness threshold
    if (bloom_enabled == 1) {
        vec3 bloom = vec3(0.0);
        float total = 0.0;
        
        const float PI2 = 6.283185;
        const int DIR_SAMPLES = 8;
        // Gaussian-ish weights for 2 rings
        const float w1 = 1.0;   // inner ring
        const float w2 = 0.4;   // outer ring
        
        // Ring 1: radius 4px
        for (int i = 0; i < DIR_SAMPLES; i++) {
            float angle = float(i) * PI2 / float(DIR_SAMPLES);
            vec2 off = vec2(cos(angle), sin(angle)) * texel_size * 4.0;
            vec3 samp = texture(scene_tex, uv + off).rgb;
            float brightness = dot(samp, vec3(0.2126, 0.7152, 0.0722));
            float contrib = max(brightness - 0.45, 0.0);
            bloom += samp * contrib * w1;
            total += w1;
        }
        // Ring 2: radius 10px, rotated 22.5° for better coverage
        for (int i = 0; i < DIR_SAMPLES; i++) {
            float angle = (float(i) + 0.5) * PI2 / float(DIR_SAMPLES);
            vec2 off = vec2(cos(angle), sin(angle)) * texel_size * 10.0;
            vec3 samp = texture(scene_tex, uv + off).rgb;
            float brightness = dot(samp, vec3(0.2126, 0.7152, 0.0722));
            float contrib = max(brightness - 0.45, 0.0);
            bloom += samp * contrib * w2;
            total += w2;
        }
        
        bloom /= max(total, 1.0);
        
        float self_bright = max(dot(color, vec3(0.2126, 0.7152, 0.0722)) - 0.4, 0.0);
        bloom += color * self_bright * 0.25;
        
        color += bloom * bloom_strength * 5.0;
    }
    
    // Edge glow — detect edges via color difference with neighbors
    if (edge_glow == 1) {
        vec3 c_up    = texture(scene_tex, uv + vec2(0.0, texel_size.y * 2.0)).rgb;
        vec3 c_down  = texture(scene_tex, uv - vec2(0.0, texel_size.y * 2.0)).rgb;
        vec3 c_left  = texture(scene_tex, uv - vec2(texel_size.x * 2.0, 0.0)).rgb;
        vec3 c_right = texture(scene_tex, uv + vec2(texel_size.x * 2.0, 0.0)).rgb;
        float edge = length(c_up - c_down) + length(c_left - c_right);
        edge = clamp(edge * edge_glow_strength, 0.0, 1.0);
        // Glow color: white-cyan
        vec3 glow_color = mix(vec3(0.3, 0.9, 1.0), vec3(1.0, 1.0, 1.0), edge);
        color += glow_color * edge * 0.6;
    }
    
    fragColor = vec4(color, 1.0);
}
"""

# =============================
# GPU Frustum Culling Compute Shader
# =============================

FRUSTUM_CULL_COMPUTE_SHADER = """
#version 430

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputPoints {
    float in_data[];
};

layout(std430, binding = 1) writeonly buffer OutputPoints {
    float out_data[];
};

layout(std430, binding = 2) buffer Counter {
    uint out_count;
};

uniform int num_points;
uniform vec4 planes[6];
uniform float margin;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(num_points)) return;

    uint base = idx * 6u;
    vec3 pos = vec3(in_data[base], in_data[base + 1u], in_data[base + 2u]);

    // Frustum test: check all 6 planes
    bool inside = true;
    for (int i = 0; i < 6; i++) {
        if (dot(planes[i].xyz, pos) + planes[i].w < -margin) {
            inside = false;
            break;
        }
    }

    if (inside) {
        uint out_idx = atomicAdd(out_count, 1u);
        uint out_base = out_idx * 6u;
        for (uint j = 0u; j < 6u; j++) {
            out_data[out_base + j] = in_data[base + j];
        }
    }
}
"""

# =============================
# GPU-driven Point Cloud Build Compute Shader
# depth texture → compute → SSBO → indirect draw
# Eliminates: CPU numpy, VBO upload, readback stall
# =============================

POINTCLOUD_BUILD_COMPUTE_SHADER = """
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D depth_tex;       // R32F, depth in millimetres (current frame)
uniform sampler2D color_tex;       // RGB8 — data is BGR, swizzle in shader
uniform sampler2D prev_depth_tex;  // R32F, depth in millimetres (previous Kinect frame)
uniform sampler2D prev_color_tex;  // RGB8 — data is BGR, swizzle in shader

// Persistent temporal-smoothing depth state (mm, read-write)
layout(r32f, binding = 0) uniform image2D temporal_depth_img;

// Output point cloud: interleaved (x, y, z, r, g, b) per point
layout(std430, binding = 0) writeonly buffer OutputPoints {
    float out_data[];
};

// Indirect draw command — first uint is the vertex count
// Layout matches GL DrawArraysIndirectCommand
layout(std430, binding = 1) buffer DrawIndirect {
    uint draw_count;
    uint draw_instance_count;
    uint draw_first;
    uint draw_base_instance;
};

uniform int cloud_w, cloud_h;

// Depth camera intrinsics (pre-scaled to cloud resolution)
uniform float fx_d, fy_d, cx_d, cy_d;
// Color camera intrinsics (pre-scaled to cloud resolution)
uniform float fx_c, fy_c, cx_c, cy_c;
// Stereo baseline (meters)
uniform float baseline_x, baseline_y;

uniform float depth_min, depth_max;
uniform int depth_color_align;

uniform int frustum_culling_on;
uniform vec4 frustum_planes[6];
uniform float frustum_margin;
uniform uint max_points;

uniform int cluster_coloring_on;
uniform float cluster_blend;
uniform float cluster_threshold;  // normalized 0..1

uniform int edge_filter_on;
uniform float edge_filter_threshold;  // meters — max allowed depth jump to neighbors

// Interpolation / temporal smoothing (GPU-side)
uniform float interp_alpha;              // 0 = no interpolation, >0 = blend prev→curr
uniform int   interp_has_prev;           // 1 if prev textures contain valid data
uniform float interp_snap_threshold_mm;  // 200.0 mm — snap instead of blend
uniform int   temporal_on;               // 1 = temporal smoothing enabled
uniform float temporal_alpha;            // EMA alpha (0.05..1.0)
uniform float temporal_snap_threshold_mm; // mm — snap threshold for temporal

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= cloud_w || gid.y >= cloud_h) return;

    // ── Sample current depth (mm) ──
    float z_curr_mm = texelFetch(depth_tex, gid, 0).r;

    // ── Interpolation (30 Hz → 60 Hz blending) ──
    float z_mm = z_curr_mm;
    float color_snap = 1.0;  // 1.0 = use curr color only

    if (interp_has_prev == 1 && interp_alpha > 0.0) {
        float z_prev_mm = texelFetch(prev_depth_tex, gid, 0).r;
        if (z_prev_mm > 0.0 && z_curr_mm > 0.0) {
            float snap = step(interp_snap_threshold_mm, abs(z_curr_mm - z_prev_mm));
            z_mm = mix(mix(z_prev_mm, z_curr_mm, interp_alpha), z_curr_mm, snap);
            color_snap = snap;
        }
    }

    // ── Temporal smoothing (mm space) ──
    if (temporal_on == 1) {
        float z_temp = imageLoad(temporal_depth_img, gid).r;
        if (z_temp > 0.0 && z_mm > 0.0) {
            float snap_t = step(temporal_snap_threshold_mm, abs(z_mm - z_temp));
            z_temp = mix(mix(z_temp, z_mm, temporal_alpha), z_mm, snap_t);
            imageStore(temporal_depth_img, gid, vec4(z_temp, 0, 0, 0));
            z_mm = z_temp;
        } else {
            imageStore(temporal_depth_img, gid, vec4(z_mm, 0, 0, 0));
        }
    }

    // ── Convert mm → metres ──
    float z = z_mm * 0.001;

    // Range filter
    if (z <= depth_min || z >= depth_max) return;

    // Edge filter: discard flying pixels at depth discontinuities
    if (edge_filter_on == 1) {
        // Use texelFetch (integer coords, no filtering) — faster than texture()
        float zL = (gid.x > 0)          ? texelFetch(depth_tex, gid + ivec2(-1,  0), 0).r * 0.001 : z;
        float zR = (gid.x < cloud_w-1)  ? texelFetch(depth_tex, gid + ivec2( 1,  0), 0).r * 0.001 : z;
        float zU = (gid.y > 0)          ? texelFetch(depth_tex, gid + ivec2( 0, -1), 0).r * 0.001 : z;
        float zD = (gid.y < cloud_h-1)  ? texelFetch(depth_tex, gid + ivec2( 0,  1), 0).r * 0.001 : z;
        float maxDiff = max(max(abs(z - zL), abs(z - zR)),
                           max(abs(z - zU), abs(z - zD)));
        if (maxDiff > edge_filter_threshold) return;
    }

    // Unproject depth pixel → 3D position
    float px = float(gid.x);
    float py = float(gid.y);
    vec3 pos = vec3(
        (px - cx_d) * z / fx_d,
        -((py - cy_d) * z / fy_d),
        -z
    );

    // Frustum culling (6-plane test)
    if (frustum_culling_on == 1) {
        for (int i = 0; i < 6; i++) {
            if (dot(frustum_planes[i].xyz, pos) + frustum_planes[i].w < -frustum_margin)
                return;
        }
    }

    // Color UV — optionally correct for IR↔RGB parallax + different FOV
    vec2 cuv;
    vec2 uv = (vec2(gid) + 0.5) / vec2(cloud_w, cloud_h);
    if (depth_color_align == 1) {
        float zs = max(z, 0.3);
        float cu = (fx_c / fx_d) * (px - cx_d) + cx_c + fx_c * baseline_x / zs;
        float cv = (fy_c / fy_d) * (py - cy_d) + cy_c + fy_c * baseline_y / zs;
        cuv = (vec2(cu, cv) + 0.5) / vec2(cloud_w, cloud_h);
    } else {
        cuv = uv;
    }
    cuv = clamp(cuv, vec2(0.0), vec2(1.0));

    // Color sampling with interpolation blending (BGR→RGB swizzle)
    vec3 c_curr = texture(color_tex, cuv).bgr;
    vec3 color;
    if (interp_has_prev == 1 && interp_alpha > 0.0 && color_snap < 0.5) {
        vec3 c_prev = texture(prev_color_tex, cuv).bgr;
        color = mix(c_prev, c_curr, interp_alpha);
    } else {
        color = c_curr;
    }

    // Cluster coloring (depth-based near/far tinting)
    if (cluster_coloring_on == 1) {
        float rng = max(depth_max - depth_min, 0.001);
        float t = clamp((z - depth_min) / rng, 0.0, 1.0);
        float is_near = 1.0 - step(cluster_threshold, t);
        vec3 near_tint = vec3(0.15, 0.85, 0.95);
        vec3 far_tint  = vec3(0.95, 0.30, 0.20);
        vec3 tint = mix(far_tint, near_tint, is_near);
        color = color * (1.0 - cluster_blend) + tint * cluster_blend;
    }

    // Atomic append → SSBO + indirect draw count (zero CPU sync)
    uint idx = atomicAdd(draw_count, 1u);
    if (idx >= max_points) return;

    uint base = idx * 6u;
    out_data[base + 0u] = pos.x;
    out_data[base + 1u] = pos.y;
    out_data[base + 2u] = pos.z;
    out_data[base + 3u] = color.r;
    out_data[base + 4u] = color.g;
    out_data[base + 5u] = color.b;
}
"""

# =============================
# GPU Ghost Particles Compute Shader
# CPU effect → GPU transfer
# Double-buffered ghost history SSBO
# =============================

GHOST_PARTICLES_COMPUTE_SHADER = """
#version 430

layout(local_size_x = 256) in;

// Main point cloud output (shared with build shader, read-write)
layout(std430, binding = 0) buffer OutputPoints {
    float point_data[];
};

// Indirect draw buffer (vertex count in first uint)
layout(std430, binding = 1) buffer DrawIndirect {
    uint draw_count;
    uint draw_instance_count;
    uint draw_first;
    uint draw_base_instance;
};

// Ghost history: read (previous frame)
layout(std430, binding = 2) readonly buffer GhostIn {
    float ghost_in[];   // 7 floats per entry: x, y, z, r, g, b, ttl
};

// Ghost history: write (alive old ghosts + new samples)
layout(std430, binding = 3) buffer GhostOut {
    float ghost_out[];
};

// Ghost output counter
layout(std430, binding = 4) buffer GhostCounter {
    uint ghost_out_count;
};

uniform uint ghost_in_count;       // ghost entries from previous frame
uniform float ghost_sample_rate;   // GHOST_SAMPLE (0.08)
uniform int ghost_ttl;             // GHOST_TTL (14)
uniform uint max_points;
uniform uint max_ghost_points;
uniform float frame_seed;
uniform uint max_dispatch;          // dispatch upper bound for phase 1

// Phase control: dispatch Phase 0 and Phase 1 SEPARATELY with a
// glMemoryBarrier between them to prevent a data race.
uniform int phase;  // 0 = decay old ghosts, 1 = sample new points

// Shared memory for batched atomic reservation (reduces global atomic
// contention from N threads → N/256 workgroups — ~256x less contention)
shared uint s_draw_wants;       // how many threads in this WG need a draw slot
shared uint s_draw_base;        // global base index reserved for this WG
shared uint s_ghost_wants;      // how many threads need a ghost_out slot
shared uint s_ghost_base;       // global base index for ghost_out
shared uint s_local_draw_idx;   // per-WG sub-allocator for draw slots
shared uint s_local_ghost_idx;  // per-WG sub-allocator for ghost_out slots

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    // Phase 0: Decay old ghosts — append faded to output, copy alive to new history
    if (phase == 0) {
        // ── Step 1: each thread decides if it needs slots ──
        bool alive = false;
        bool needs_draw = false;
        bool needs_ghost = false;
        float fade = 0.0;
        float new_ttl = 0.0;
        uint gbase = 0u;

        if (idx < ghost_in_count) {
            gbase = idx * 7u;
            float ttl = ghost_in[gbase + 6u];
            if (ttl > 0.0) {
                alive = true;
                fade = (ttl / float(ghost_ttl)) * 0.8;
                new_ttl = ttl - 1.0;
                needs_draw = true;
                needs_ghost = (new_ttl > 0.0);
            }
        }

        // ── Step 2: count slots needed in this workgroup ──
        if (lid == 0u) {
            s_draw_wants = 0u;
            s_ghost_wants = 0u;
        }
        barrier();

        uint my_draw_local = 0u;
        uint my_ghost_local = 0u;
        if (needs_draw)
            my_draw_local = atomicAdd(s_draw_wants, 1u);
        if (needs_ghost)
            my_ghost_local = atomicAdd(s_ghost_wants, 1u);
        barrier();

        // ── Step 3: one thread per WG does the global atomicAdd ──
        if (lid == 0u) {
            if (s_draw_wants > 0u)
                s_draw_base = atomicAdd(draw_count, s_draw_wants);
            else
                s_draw_base = 0u;
            if (s_ghost_wants > 0u)
                s_ghost_base = atomicAdd(ghost_out_count, s_ghost_wants);
            else
                s_ghost_base = 0u;
        }
        barrier();

        // ── Step 4: write data using local offsets ──
        if (needs_draw) {
            uint out_idx = s_draw_base + my_draw_local;
            if (out_idx < max_points) {
                uint obase = out_idx * 6u;
                point_data[obase + 0u] = ghost_in[gbase + 0u];
                point_data[obase + 1u] = ghost_in[gbase + 1u];
                point_data[obase + 2u] = ghost_in[gbase + 2u];
                point_data[obase + 3u] = clamp(ghost_in[gbase + 3u] * fade, 0.0, 1.0);
                point_data[obase + 4u] = clamp(ghost_in[gbase + 4u] * fade, 0.0, 1.0);
                point_data[obase + 5u] = clamp(ghost_in[gbase + 5u] * fade, 0.0, 1.0);
            }
        }
        if (needs_ghost) {
            uint new_idx = s_ghost_base + my_ghost_local;
            if (new_idx < max_ghost_points) {
                uint nbase = new_idx * 7u;
                for (uint j = 0u; j < 6u; j++)
                    ghost_out[nbase + j] = ghost_in[gbase + j];
                ghost_out[nbase + 6u] = new_ttl;
            }
        }
    }

    // Phase 1: Sample current frame points → new ghost history
    if (phase == 1) {
        if (idx >= max_dispatch) return;
        uint pc = draw_count;
        if (idx >= pc) return;

        uint seed = pcg(idx ^ floatBitsToUint(frame_seed));
        float rnd = float(seed) / 4294967295.0;

        // ── Batched atomic for Phase 1 ghost_out ──
        bool wants_ghost = (rnd < ghost_sample_rate);

        if (lid == 0u) {
            s_ghost_wants = 0u;
        }
        barrier();

        uint my_local = 0u;
        if (wants_ghost)
            my_local = atomicAdd(s_ghost_wants, 1u);
        barrier();

        if (lid == 0u) {
            if (s_ghost_wants > 0u)
                s_ghost_base = atomicAdd(ghost_out_count, s_ghost_wants);
            else
                s_ghost_base = 0u;
        }
        barrier();

        if (wants_ghost) {
            uint new_idx = s_ghost_base + my_local;
            if (new_idx < max_ghost_points) {
                uint sbase = idx * 6u;
                uint nbase = new_idx * 7u;
                for (uint j = 0u; j < 6u; j++)
                    ghost_out[nbase + j] = point_data[sbase + j];
                ghost_out[nbase + 6u] = float(ghost_ttl);
            }
        }
    }
}
"""

# =============================
# GPU Particle Trails Compute Shader
# CPU effect → GPU transfer
# Double-buffered trail history SSBO
# =============================

PARTICLE_TRAILS_COMPUTE_SHADER = """
#version 430

layout(local_size_x = 256) in;

// Main point cloud output (append drip points here)
layout(std430, binding = 0) buffer OutputPoints {
    float point_data[];
};

// Indirect draw buffer
layout(std430, binding = 1) buffer DrawIndirect {
    uint draw_count;
    uint draw_instance_count;
    uint draw_first;
    uint draw_base_instance;
};

// Trail base points: read (previous frame)
layout(std430, binding = 2) readonly buffer TrailIn {
    float trail_in[];   // 8 floats: x, y, z, r, g, b, born_time, lifetime
};

// Trail base points: write (alive old + new frozen)
layout(std430, binding = 3) buffer TrailOut {
    float trail_out[];
};

// Trail output counter
layout(std430, binding = 4) buffer TrailCounter {
    uint trail_out_count;
};

uniform uint trail_in_count;
uniform float current_time;
uniform int freeze_this_frame;
uniform uint n_freeze;
uniform float freeze_lifetime;
uniform uint max_points;
uniform uint max_trail_points;
uniform float frame_seed;
uniform uint max_dispatch;

uniform int phase;  // 0 = process existing trails, 1 = freeze new trails

// Shared memory for batched atomic reservation
shared uint s_trail_wants;
shared uint s_trail_base;
shared uint s_draw_wants;
shared uint s_draw_base;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    // Phase 0: Process existing trail points — alive check + drip rendering
    if (phase == 0) {
        bool alive = false;
        bool needs_trail = false;
        uint my_drip_count = 0u;
        float progress = 0.0;
        float fade_val = 0.0;
        uint tbase = 0u;

        if (idx < trail_in_count) {
            tbase = idx * 8u;
            float born = trail_in[tbase + 6u];
            float lifetime = trail_in[tbase + 7u];
            float age = current_time - born;
            if (age < lifetime) {
                alive = true;
                needs_trail = true;
                progress = age / lifetime;
                float drip_len = progress * 0.6;
                my_drip_count = uint(max(int(drip_len / 0.015), 1)) + 1u;
                fade_val = max(1.0 - progress * 0.8, 0.15);
            }
        }

        // ── Batch trail_out_count atomic ──
        if (lid == 0u) {
            s_trail_wants = 0u;
            s_draw_wants = 0u;
        }
        barrier();

        uint my_trail_local = 0u;
        if (needs_trail)
            my_trail_local = atomicAdd(s_trail_wants, 1u);

        // Each alive thread also reserves drip slots
        uint my_draw_local = 0u;
        if (alive)
            my_draw_local = atomicAdd(s_draw_wants, my_drip_count);
        barrier();

        if (lid == 0u) {
            if (s_trail_wants > 0u)
                s_trail_base = atomicAdd(trail_out_count, s_trail_wants);
            else
                s_trail_base = 0u;
            if (s_draw_wants > 0u)
                s_draw_base = atomicAdd(draw_count, s_draw_wants);
            else
                s_draw_base = 0u;
        }
        barrier();

        // Write trail history
        if (needs_trail) {
            uint new_idx = s_trail_base + my_trail_local;
            if (new_idx < max_trail_points) {
                uint nbase = new_idx * 8u;
                for (uint j = 0u; j < 8u; j++)
                    trail_out[nbase + j] = trail_in[tbase + j];
            }
        }

        // Write drip points
        if (alive) {
            uint out_start = s_draw_base + my_draw_local;
            int n_steps = int(my_drip_count) - 1;

            float bx = trail_in[tbase + 0u];
            float by = trail_in[tbase + 1u];
            float bz = trail_in[tbase + 2u];
            float br = trail_in[tbase + 3u];
            float bg = trail_in[tbase + 4u];
            float bb = trail_in[tbase + 5u];

            for (int s = 0; s <= n_steps; s++) {
                uint oidx = out_start + uint(s);
                if (oidx >= max_points) break;

                float sf = fade_val * max(1.0 - float(s) / float(n_steps + 1), 0.08);
                uint obase = oidx * 6u;
                point_data[obase + 0u] = bx;
                point_data[obase + 1u] = by - float(s) * 0.015;
                point_data[obase + 2u] = bz;
                point_data[obase + 3u] = clamp(br * sf, 0.0, 1.0);
                point_data[obase + 4u] = clamp(bg * sf, 0.0, 1.0);
                point_data[obase + 5u] = clamp(bb * sf, 0.0, 1.0);
            }
        }
    }

    // Phase 1: Freeze new trail points from current frame
    if (phase == 1) {
        if (idx >= max_dispatch) return;
        if (freeze_this_frame == 0) return;
        uint pc = draw_count;
        if (idx >= pc) return;

        uint seed = pcg(idx ^ floatBitsToUint(frame_seed) ^ 0xBEEFCAFEu);
        float rnd = float(seed) / 4294967295.0;
        float prob = float(n_freeze) / float(max(pc, 1u));

        bool wants_trail = (rnd < prob);

        if (lid == 0u) {
            s_trail_wants = 0u;
        }
        barrier();

        uint my_local = 0u;
        if (wants_trail)
            my_local = atomicAdd(s_trail_wants, 1u);
        barrier();

        if (lid == 0u) {
            if (s_trail_wants > 0u)
                s_trail_base = atomicAdd(trail_out_count, s_trail_wants);
            else
                s_trail_base = 0u;
        }
        barrier();

        if (wants_trail) {
            uint new_idx = s_trail_base + my_local;
            if (new_idx < max_trail_points) {
                uint sbase = idx * 6u;
                uint nbase = new_idx * 8u;
                for (uint j = 0u; j < 6u; j++)
                    trail_out[nbase + j] = point_data[sbase + j];
                trail_out[nbase + 6u] = current_time;
                trail_out[nbase + 7u] = freeze_lifetime;
            }
        }
    }
}
"""
