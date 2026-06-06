# flux
A testing repository for FEM development and learning about solving different meshing problems in multiple dimensions.

I am currently developing a transient 2D thermal model to simulate furnace chamber performance and calculate ideal heater powers, though I intend to expand this solver to handle arbitrary meshes and different FEM problems such as fluid flow and FEA.

Flux is is a transient, two-dimensional heat transfer solver that I'm writing to learn more about writing parallelised finite-element-analysis simulations. Flux can be used to simulate thermal environments consisting of multiple interacting meshes, lumped capacitors (constant temperature), and environmental objects, each definable by materials. I started to develop Flux to simulate the thermal environment within my home made metal casting furnace, [OctoForge](https://github.com/DeltaFwulf/OctoForge). This is a challenging environment, with a cylindrical geometry and multimodel heat transfer between several pairs of objects.

Meshes and Lumped Capacitors are bounded by rectilinear lines; notches and holes are supported. While lumped capacitors have constant internal properties, meshes contain internal mesh nodes that are solved using an ADI technique at each timestep.

There exists a plotting suite for visualising results, such as the net flux/power/energy at different sets of edges; edges can be grouped into named sets to aid in analysis. For further post-processing, results may also be saved to file, allowing the user to reload quickly a long run and change plot settings.
<br>
<br>
# Gallery
Here are a few outputs (please note, this project is still **very much** in development, and results may not be entirely accurate.
<br>

<br>
This image shows the transient temperature of one of OctoForge's firebricks when radiatively heated by elements at 1400K in each notch face. The outer face convects to ambient air at 288 K, and all other faces are insulated.
<img width="640" height="480" alt="firebrick_20_minutes" src="https://github.com/user-attachments/assets/6338884f-3651-4534-a647-58004736ecfb" />
<br>

<br>
The image below shows two rectangular prisms convecting heat away on all exposed sides, and conducting into one another. The meshes start at 500 and 600 K, respectively, and the air is at 288 K.
_Flux_ calculates the appropriate convection correlation to use (natural convection, vertical, or stable/unstable horizontal wall, or forced if above critical Reynolds number).
<img width="640" height="480" alt="flux_animation" src="https://github.com/user-attachments/assets/70901c06-c545-4880-9a56-011a27f7627b" />
<br>

<br>
Here, a pipe of ID 200mm and OD 1000mm is held at 500K on the internal face and 300K on the outer face, with the top and bottom faces insulated (infinite length condition). The pipe tends to teh natural logarithm curve for steady state cylindrical objects.

<img width="640" height="480" alt="pipe_30k" src="https://github.com/user-attachments/assets/4923988c-2434-4920-aa4e-9fbce5a3533e" />
