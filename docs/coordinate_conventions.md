# CLEAR360 — Coordinate Conventions & Mathematical Reference

This document formalises all spatial reference frames, transformations, camera projection models, depth representations, and file encoding conventions used throughout the CLEAR360 project.

---

## 1. World / Vehicle Reference Frame

CLEAR360 defines a single global coordinate system aligned with the host vehicle in its nominal resting pose on a flat horizontal plane.

- **Origin $(0, 0, 0)$**: Vehicle geometric ground centre projected onto the road surface $Z = 0$ (or vehicle center of volume projected to ground).
- **$X$-axis**: Points to the **RIGHT** of the vehicle (lateral axis, transverse).
- **$Y$-axis**: Points **FORWARD** along the primary vehicle heading (longitudinal axis).
- **$Z$-axis**: Points **UP** orthogonal to the ground plane (vertical axis).
- **Handedness**: Right-handed Cartesian system ($X \times Y = Z$).

### Vehicle Geometry Reference (Hyundai Creta 2022-approximate)
- Length: $4.30\text{ m}$ (spanning $Y \in [-2.15, +2.15]\text{ m}$). Front bumper is at $Y \approx +2.15\text{ m}$, rear bumper at $Y \approx -2.15\text{ m}$.
- Width: $1.79\text{ m}$ (spanning $X \in [-0.895, +0.895]\text{ m}$).
- Height: $1.635\text{ m}$ (spanning $Z \in [0.0, 1.635]\text{ m}$).
- Wheelbase: $2.60\text{ m}$ (Front axle at $Y = +1.30\text{ m}$, Rear axle at $Y = -1.30\text{ m}$).

---

## 2. Camera Frame (OpenCV Standard)

For multi-view geometry, AI depth estimation, and 3D point cloud unprojection, CLEAR360 adheres strictly to the standard **OpenCV camera reference frame**:

- **Origin $(0, 0, 0)$**: Camera optical centre (pinhole focal point).
- **$+x_{\text{cam}}$**: Points to the **right** across the image plane (increasing column index $u$).
- **$+y_{\text{cam}}$**: Points **downwards** across the image plane (increasing row index $v$).
- **$+z_{\text{cam}}$**: Points **forward** into the 3D scene along the principal optical axis (depth direction).
- **Handedness**: Right-handed ($x_{\text{cam}} \times y_{\text{cam}} = z_{\text{cam}}$).

---

## 3. Blender Camera Frame

Blender uses a different internal convention for its virtual camera objects:
- **$+x_{\text{bl}}$**: Points right across the camera viewport.
- **$+y_{\text{bl}}$**: Points **upwards** across the camera viewport.
- **$+z_{\text{bl}}$**: Points **backwards** away from the scene; the camera looks down its local **$-z_{\text{bl}}$** axis.

---

## 4. Transformation: Blender Camera-to-World $\to$ OpenCV Camera-to-World

Let $M_{\text{bl}} \in \mathbb{R}^{4 \times 4}$ denote the Blender object matrix in world space (`cam_obj.matrix_world`), representing the Blender camera-to-world transform:
$$
\mathbf{P}_{\text{world}} = M_{\text{bl}} \, \mathbf{P}_{\text{bl}}
$$
Because the OpenCV camera frame has:
- $x_{\text{cv}} = x_{\text{bl}}$
- $y_{\text{cv}} = -y_{\text{bl}}$
- $z_{\text{cv}} = -z_{\text{bl}}$

The conversion matrix $C$ from OpenCV camera coordinates to Blender camera coordinates is:
$$
C = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} = \operatorname{diag}(1, -1, -1, 1)
$$
Therefore, the **OpenCV Camera-to-World matrix** $T_{\text{c2w}}^{\text{cv}}$ is:
$$
T_{\text{c2w}}^{\text{cv}} = M_{\text{bl}} \cdot \operatorname{diag}(1, -1, -1, 1)
$$
And the **World-to-Camera (extrinsics) matrix** $T_{\text{w2c}}^{\text{cv}}$ (stored as `T_cam_world` in `cameras.yaml`) is:
$$
T_{\text{w2c}}^{\text{cv}} = \left( T_{\text{c2w}}^{\text{cv}} \right)^{-1} = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$
where $R \in \operatorname{SO}(3)$ and $\mathbf{t} \in \mathbb{R}^3$.

---

## 5. Extrinsic Matrix $T_{\text{cam\_world}}$

The extrinsic matrix maps 3D points expressed in the world/vehicle frame $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^T$ into the OpenCV camera reference frame $\mathbf{P}_c = [x_c, y_c, z_c, 1]^T$:

$$
\mathbf{P}_c = T_{\text{cam\_world}} \, \mathbf{P}_w = \begin{bmatrix} R_{11} & R_{12} & R_{13} & t_x \\ R_{21} & R_{22} & R_{23} & t_y \\ R_{31} & R_{32} & R_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

In `cameras.yaml`, `T_cam_world` is serialized as a $4 \times 4$ row-major nested list.

---

## 6. Intrinsic Matrix $K$

Under the idealized pinhole camera model without lens distortion:
$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$
- **Image Resolution**: $W = 1280$, $H = 720$.
- **Principal Point**: $(c_x, c_y) = (W / 2, H / 2) = (640.0, 360.0)$.
- **Pixel Coordinates**: $(u, v)$ with $(0, 0)$ at the top-left corner of the image.
  - $u \in [0, W-1]$ (horizontal, column index)
  - $v \in [0, H-1]$ (vertical, row index)
- **Focal Length**:
  Given sensor width $S_w = 36.0\text{ mm}$ and physical focal length $f_{\text{mm}} = 10.3923\text{ mm}$:
  $$
  f_x = f_y = \frac{f_{\text{mm}}}{S_w} \cdot W = \frac{10.3923}{36.0} \cdot 1280 \approx 369.5\text{ pixels}
  $$
  Equivalently from the horizontal field of view $\theta_h \approx 120^\circ$:
  $$
  f_x = \frac{W}{2 \tan(\theta_h / 2)} = \frac{1280}{2 \tan(60^\circ)} = \frac{640}{\sqrt{3}} \approx 369.504\text{ pixels}
  $$

---

## 7. 3D Unprojection Formula

Given pixel coordinates $(u, v)$ and measured camera Z-depth $d_z = z_c$:

### Step 1: Pixel to Camera Frame
$$
x_c = \frac{u - c_x}{f_x} \cdot d_z
$$
$$
y_c = \frac{v - c_y}{f_y} \cdot d_z
$$
$$
z_c = d_z
$$
In vector notation:
$$
\mathbf{p}_c = \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} = d_z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

### Step 2: Camera Frame to World Frame
Using $T_{\text{c2w}} = \left( T_{\text{cam\_world}} \right)^{-1}$:
$$
\begin{bmatrix} \mathbf{P}_w \\ 1 \end{bmatrix} = T_{\text{c2w}} \begin{bmatrix} \mathbf{p}_c \\ 1 \end{bmatrix} = \begin{bmatrix} R_{\text{c2w}} & \mathbf{t}_{\text{c2w}} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} \mathbf{p}_c \\ 1 \end{bmatrix}
$$
$$
\mathbf{P}_w = R_{\text{c2w}} \, \mathbf{p}_c + \mathbf{t}_{\text{c2w}}
$$

---

## 8. Depth Conventions: Z-Depth vs. Euclidean Distance

- **Camera Z-Depth ($d_z$)**:
  The orthogonal distance along the camera optical axis $+z_{\text{cam}}$:
  $$
  d_z = z_c
  $$
  Blender's `RenderLayers.Depth` pass outputs $d_z$ in metres.
- **Euclidean Distance ($d_e$)**:
  The 3D straight-line distance from the optical center to the 3D surface point:
  $$
  d_e = \|\mathbf{p}_c\|_2 = \sqrt{x_c^2 + y_c^2 + z_c^2} = d_z \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}
  $$

| Context | Preferred Representation | Reason |
|---------|--------------------------|--------|
| Blender Ground Truth (`gt_depth/`) | Z-depth $d_z$ | Standard pinhole projection linear relationship with $K^{-1}$ |
| AI Monocular Estimator | Z-depth $d_z$ | Aligned with standard training targets |
| LiDAR / Range Sensors | Euclidean $d_e$ | Physical time-of-flight measurement |

---

## 9. Semantic Pass Indices

Each object in the Blender simulation is assigned a distinct `pass_index` mapped into ground-truth semantic mask PNGs:

| `pass_index` | Semantic Class | Description |
|:---:|---|---|
| 0 | `unknown / background` | Unlabeled scene geometry, distant background |
| 1 | `road` | Ground driveable asphalt/concrete plane ($20 \times 20\text{ m}$) |
| 2 | `vehicle_ego` | Host vehicle body, roof, wheels, mirrors |
| 3 | `parked_vehicle` | Adjacent parked vehicles |
| 4 | `wall` | Boundary garage walls (North, South, East, West) |
| 5 | `pillar` | Structural support columns |
| 6 | `obstacle_dynamic` | Dynamic moving obstacle |
| 7 | `kerb_speed_breaker` | Raised kerbs, speed bumps |

---

## 10. EXR File Format and Channel Specifications

All EXR files are encoded as single-precision 32-bit floats (`FLOAT` pixel type):

| Directory | Format | Channels | Content |
|---|---|---|---|
| `data/gt_depth/{camera}/frame_{n:04d}.exr` | OpenEXR | `Z` | Orthogonal camera Z-depth in metres ($d_z \ge 0$) |
| `data/gt_position/{camera}/frame_{n:04d}.exr` | OpenEXR | `X`, `Y`, `Z` (or `R`, `G`, `B`) | Absolute world-frame coordinates $(X_w, Y_w, Z_w)$ in metres |
| `data/sim_frames/{camera}/frame_{n:04d}.png` | PNG | `RGB` | 8-bit per channel color render |
| `data/gt_semantic/{camera}/frame_{n:04d}.png` | PNG | `L` (Grayscale) | 8-bit integer containing `pass_index` $(0 - 7)$ |
