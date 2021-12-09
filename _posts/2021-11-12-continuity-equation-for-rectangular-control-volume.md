---
layout: post
title: Derivation of continuity equation for a rectangular control space
tags: ["computational acoustic","fluid dynamics","continuity equation"]
mathjax: true
---

**Credit**: This post aims to document the derivation of the continuity equation in fluid dynamics. It is motivated from the following [YouTube video](https://www.youtube.com/watch?v=Ls5HS2MLXpg).  Please watch the video to gain a greater understanding.


**Continuity Equation**: 

The continuity equation states the conservation of mass in continuum mechanics analysis. The equation can be derived by considering the fluid mass flow rate in and out of an infinitesimally small controlled space (volume) and the rate of change of the fluid mass inside the controlled space. Here, we assume a rectangular control space. But this can be extended for other geometries.

**Parameters**:

$$ \dot{m}_{in} = \text{rate of mass flow into the control space} $$

$$ \dot{m}_{out} = \text{rate of mass flow out of the control space} $$

$$ \rho = \text{density of the fluid} $$

$$ u = \text{flow velocity or particle velocity}$$

$$u_x, u_y, u_z = \text{particle velocity along the x, y and z directions respectively}$$

$$ A = \text{surface area of the control space}$$

$$ \text{Q} = \frac{\partial V}{\partial t} = u.A = \text{volumetric flow rate or volume velocity}$$

**Derivation**:

According to the law of conservation of mass,<br/>
rate of mass change within the control space = mass flow rate into the space - mass flow rate out of the space, i.e.,

$$ 
\begin{align*}
\frac{\partial m}{\partial t} &=  \dot{m}_{in} - \dot{m}_{out} \\

&= (\rho Q_x + \rho Q_y + \rho Q_z) - (\rho Q_{x+\Delta{x}} + \rho Q_{y+\Delta{y}} + \rho Q_{z+\Delta{z}})
\end{align*}
$$

The fluid mass in the control space can be defined as follows,

$$ 
\begin{align*}
\text{mass} &= \text{density} \times \text{volume} \\

m &= \rho \partial x \partial y \partial z
\end{align*}
$$

if we replace fliud mass in the main equation, one obtains

$$
\begin{align*}
\frac{\partial (\rho \partial x \partial y \partial z)}{\partial t} &= (\rho u_x \partial y \partial z + \rho u_y \partial x \partial z + \rho u_z \partial x \partial y) - (\rho u_{x+\Delta{x}} \partial y \partial z + \rho u_{y+\Delta{y}} \partial x \partial z + \rho u_{z+\Delta{z}} \partial x \partial y) \\

\frac{\partial  \rho}{\partial t} &= \frac{\rho u_x - \rho u_{x+\Delta{x}}}{\partial x} + \frac{\rho u_y - \rho u_{y+\Delta{y}}}{\partial y} + \frac{\rho u_z - \rho u_{z+\Delta{z}}}{\partial z} \\

\frac{\partial  \rho}{\partial t} &= - \frac{\partial (\rho u_x)}{\partial t} - \frac{\partial (\rho u_y)}{\partial t} - \frac{\partial (\rho u_z)}{\partial t}
\end{align*}
$$

Rearranging the above equation gives,

> $$ \frac{\partial  \rho}{\partial t} + \frac{\partial (\rho u_x)}{\partial t} + \frac{\partial (\rho u_y)}{\partial t} + \frac{\partial (\rho u_z)}{\partial t} = 0 $$

> $$ \frac{\partial  \rho}{\partial t} + \nabla (\rho u) = 0 $$ 

These are the complete and most general forms of the continuity equation that enforces conservation of mass. It applies to all materials.
