import warp as wp
import newton
wp.init()
builder = newton.ModelBuilder()
body = builder.add_body(label="test")
builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=1.0))
builder.body_mass[body] = 1.0
builder.body_inertia[body] = wp.vec3(0.1, 0.2, 0.3)
model = builder.finalize(requires_grad=False)
print("Mass:", model.body_mass.numpy()[body])
print("Inertia:", model.body_inertia.numpy()[body])
