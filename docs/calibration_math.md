# CLEAR360 — Camera Calibration Mathematics

This document derives the camera projection models, intrinsic and extrinsic parameter formulations, field-of-view relationships, and lens distortion principles for the CLEAR360 surround-view camera system.

---

## 1. Pinhole Camera Model

A pinhole camera projects 3D points $\mathbf{P}_c = (x_c, y_c, z_c)^T$ in the camera coordinate frame onto a 2D image plane at focal distance $f$:

$$
\tilde{\mathbf{p}} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \, \mathbf{P}_c
$$

where $\sim$ denotes equality up to scale (homogeneous coordinates):

$$
z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix}
$$

Carrying out the multiplication and dividing by $z_c$:
$$
u = f_x \frac{x_c}{z_c} + c_x, \quad v = f_y \frac{y_c}{z_c} + c_y
$$

---

## 2. Intrinsic Matrix Derivation

### Focal Length in Pixels
Camera sensors specify physical focal length $f_{\text{mm}}$ in millimetres and sensor dimensions $S_w \times S_h$ in millimetres. The conversion to pixel units for image resolution $W \times H$ is:

$$
f_x = \frac{f_{\text{mm}}}{S_w} \cdot W
$$
$$
f_y = \frac{f_{\text{mm}}}{S_h} \cdot H
$$

Assuming square pixels ($f_x = f_y$):
$$
\frac{W}{S_w} = \frac{H}{S_h} \implies \text{Aspect Ratio} = \frac{W}{H} = \frac{S_w}{S_h}
$$

### Principal Point
Under zero optical decentering, the principal point $(c_x, c_y)$ coincides with the optical image center:
$$
c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}
$$

---

## 3. Horizontal and Vertical Field of View (HFOV / VFOV)

The relationship between focal length $f_x, f_y$, image size $W, H$, and angular Field of View $\theta_h, \theta_v$ is given by basic trigonometry:

$$
\tan\left(\frac{\theta_h}{2}\right) = \frac{W / 2}{f_x} \implies f_x = \frac{W}{2 \tan(\theta_h / 2)}
$$
$$
\tan\left(\frac{\theta_v}{2}\right) = \frac{H / 2}{f_y} \implies f_y = \frac{H}{2 \tan(\theta_v / 2)}
$$

Inverting these equations gives the angular spans:
$$
\theta_h = 2 \arctan\left(\frac{W}{2 f_x}\right)
$$
$$
\theta_v = 2 \arctan\left(\frac{H}{2 f_y}\right)
$$

---

## 4. Extrinsic Matrix Formulation

The extrinsic transform relates points in the world reference frame $\mathbf{P}_w$ to the camera frame $\mathbf{P}_c$:

$$
\mathbf{P}_c = R \, \mathbf{P}_w + \mathbf{t}
$$

In homogeneous coordinates ($4 \times 4$ SE(3) matrix):
$$
T_{\text{w2c}} = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$
- $R \in \operatorname{SO}(3)$ is the $3 \times 3$ orthonormal rotation matrix ($R^T R = I, \det(R) = +1$).
- $\mathbf{t} \in \mathbb{R}^3$ is the translation vector ($t_x, t_y, t_z$).
- The camera optical center in world space is given by $\mathbf{C}_w = -R^T \mathbf{t}$.
- The inverse Camera-to-World matrix is:
$$
T_{\text{c2w}} = T_{\text{w2c}}^{-1} = \begin{bmatrix} R^T & -R^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} R_{\text{c2w}} & \mathbf{C}_w \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

---

## 5. End-to-End Projection Pipeline

Given a 3D world landmark $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^T$:

1. **Rigid Body Transformation (World $\to$ Camera)**:
   $$
   \begin{bmatrix} x_c \\ y_c \\ z_c \\ 1 \end{bmatrix} = T_{\text{cam\_world}} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
   $$

2. **Normalized Camera Coordinates**:
   $$
   x_n = \frac{x_c}{z_c}, \quad y_n = \frac{y_c}{z_c}
   $$

3. **Perspective Projection (Camera $\to$ Pixel)**:
   $$
   u = f_x \cdot x_n + c_x
   $$
   $$
   v = f_y \cdot y_n + c_y
   $$

---

## 6. CLEAR360 Camera Specifications & Numeric Values

All four surround cameras share identical optical sensor characteristics:

- **Resolution**: $W = 1280\text{ px}$, $H = 720\text{ px}$
- **Sensor Width**: $S_w = 36.0\text{ mm}$
- **Sensor Height**: $S_h = 20.25\text{ mm}$ (16:9 aspect ratio)
- **Target Horizontal FOV**: $\theta_h \approx 120.0^\circ$
- **Theoretical Focal Length**:
  $$
  f_{\text{mm}} = \frac{S_w}{2 \tan(60^\circ)} = \frac{36.0}{2 \sqrt{3}} \approx 10.3923048\text{ mm}
  $$
- **Focal Length in Pixels**:
  $$
  f_x = f_y = \frac{10.3923048}{36.0} \times 1280 \approx 369.504\text{ px}
  $$
  *(In `sim/constants.py`, the nominal value is calibrated to $f_x \approx 369.4\text{ px}$)*
- **Principal Point**:
  $$
  c_x = 640.0\text{ px}, \quad c_y = 360.0\text{ px}
  $$
- **Intrinsic Matrix $K$**:
  $$
  K = \begin{bmatrix} 369.4 & 0 & 640.0 \\ 0 & 369.4 & 360.0 \\ 0 & 0 & 1 \end{bmatrix}
  $$

### Camera Positions & Mount Angles

| Camera | Position $(X, Y, Z)$ [m] | Look Direction | Downward Tilt | HFOV |
|---|---|---|---|---|
| **Front** | $(0.0, +2.15, 0.90)$ | $+Y$ (Forward) | $-8^\circ$ | $\approx 120^\circ$ |
| **Rear** | $(0.0, -2.15, 0.90)$ | $-Y$ (Backward) | $-8^\circ$ | $\approx 120^\circ$ |
| **Left** | $(-1.09, +0.30, 1.25)$ | $-X$ (Left) | $-6^\circ$ | $\approx 120^\circ$ |
| **Right** | $(+1.09, +0.30, 1.25)$ | $+X$ (Right) | $-6^\circ$ | $\approx 120^\circ$ |

---

## 7. Lens Distortion Model (Future Phase Reference)

Real-world surround cameras with $120^\circ+$ field of view exhibit significant barrel or fisheye distortion. While Milestone M1 renders ideal pinhole images, future milestones processing physical vehicle sensors will incorporate the Brown-Conrady or Kannala-Brandt distortion model.

### Brown-Conrady Model (RadTan)
Let $(x_n, y_n)$ be the normalized undistorted image coordinates, and $r^2 = x_n^2 + y_n^2$.

**Radial Distortion**:
$$
x_{\text{rad}} = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{\text{rad}} = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

**Tangential Distortion**:
$$
x_{\text{tan}} = 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)
$$
$$
y_{\text{tan}} = p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n
$$

**Distorted Normalized Coordinates**:
$$
x_d = x_{\text{rad}} + x_{\text{tan}}, \quad y_d = y_{\text{rad}} + y_{\text{tan}}
$$

**Final Distorted Pixel**:
$$
u_d = f_x \cdot x_d + c_x, \quad v_d = f_y \cdot y_d + c_y
$$

For M1 synthetic generation, distortion coefficients are set to zero:
$$
k_1 = k_2 = k_3 = p_1 = p_2 = 0
$$
