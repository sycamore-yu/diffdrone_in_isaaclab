import warp as wp
wp.init()
@wp.kernel
def test_vec(out: wp.array(dtype=wp.spatial_vector)):
    f = wp.vec3(1.0, 2.0, 3.0)
    t = wp.vec3(4.0, 5.0, 6.0)
    out[0] = wp.spatial_vector(f, t)
out = wp.zeros(1, dtype=wp.spatial_vector)
wp.launch(test_vec, dim=1, inputs=[out])
print(out.numpy()[0])
